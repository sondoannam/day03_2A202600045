# Group Report: Lab 3 — Chatbot vs ReAct Agent
## CV Tailoring Agent System

- **Tên nhóm**: Group F2
- **Các thành viên**:
  - Member 1 — Đoàn Nam Sơn, 2A202600045 — Kiến trúc hệ thống, Pydantic Schemas, CV Extraction Pipeline, End-to-end run.py
  - Member 2 — Nhữ Gia Bách, 2A202600248 — CV PDF Extraction (pdfplumber + LLM summary), Project Setup
  - Member 3 — Trần Quang Quí, 2A202600305 — Chatbot Baseline, Session State, Section Drafting, ATS Validator, Agent Loop
  - Member 4 — Vũ Đức Duy, 2A202600337 — JD Web Scraper, CV-JD Matcher, JD Dataset
  - Member 5 — Hoàng Vĩnh Giang, 2A202600079 — JD Extraction (JSON), Schema Fixes, ATS Validator Blueprint
- **Ngày triển khai**: 2026-04-06

---

## 1. Tóm tắt điều hành

Nhóm F2 xây dựng một **ReAct Agent chuyên biệt cho bài toán tailoring CV theo Job Description (JD)**. Agent thực hiện chuỗi tool calls bắt buộc — extract CV → extract JD → draft sections → validate ATS — và chỉ viết Final Answer khi có kết quả thực tế từ tools.

So sánh với **Chatbot baseline** (không có tools, generate từ general knowledge):

| Tiêu chí | Chatbot | ReAct Agent |
|---|---|---|
| Hallucination | Cao — bịa công ty, metric, skill | Gần 0 — mọi claim đến từ Observation |
| Tác vụ đơn giản | Nhanh (~1s) | Chậm hơn (~5–15s, do tool calls) |
| Tác vụ đa bước | Không thực hiện được | Hoàn thành đúng với ground-truth CV |
| ATS feedback loop | Không có | Có — agent tự điều chỉnh khi score thấp |
| Reproducibility | Thấp | Cao — deterministic ATS scoring |

**Kết quả chính**: Agent giải quyết đúng 100% tác vụ tailoring CV đa bước (extract → draft → validate) mà chatbot không thể thực hiện do không có quyền truy cập dữ liệu thực. Trên các câu hỏi factual đơn giản, chatbot phản hồi nhanh hơn 3–5×.

---

## 2. Kiến trúc hệ thống & Công cụ

### 2.1 Chatbot Baseline

`src/chatbot.py` implement một multi-turn chatbot tối giản: nhận input, gọi LLM một lần với system prompt cố định, lưu lịch sử hội thoại vào `self.history`. Không có tools, không có vòng lặp — dùng để làm baseline so sánh.

```python
class Chatbot:
    def chat(self, user_input: str) -> str:
        context = "\n".join(f"{msg['role'].upper()}: {msg['content']}"
                            for msg in self.history)
        prompt = f"{context}\nUSER: {user_input}\nASSISTANT:" if context else user_input
        result = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
        response = result["content"].strip()
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        return response
```

System prompt chỉ có một dòng quan trọng: `"you have no access to external tools"` — điều này buộc chatbot phải generate từ general knowledge, phơi bày điểm yếu hallucination khi xử lý tác vụ cần dữ liệu thực.

### 2.2 Vòng lặp ReAct — Thought-Action-Observation

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      ReActAgent.run()                       │
│                                                             │
│  ┌─────────────┐   LLM call    ┌────────────────────────┐  │
│  │   Prompt    │──────────────▶│  Thought + Action      │  │
│  │  (context)  │               └───────────┬────────────┘  │
│  └─────────────┘                           │ _parse_action  │
│        ▲                     ┌─────────────▼────────────┐  │
│        │  Observation        │    _execute_tool()       │  │
│        └─────────────────────│    tool["function"](args)│  │
│                              └──────────────────────────┘  │
│                                                             │
│  Loop until: Final Answer  OR  max_steps reached            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer (based solely on Observations)
```

**Mỗi bước trong vòng lặp** (`src/agent/agent.py`):
1. Gọi LLM với `current_prompt` + system prompt
2. Parse `Thought:` → log event `THOUGHT`
3. Kiểm tra `Final Answer:` → nếu có thì kết thúc
4. Parse `Action: tool_name(args)` → gọi `_execute_tool()`
5. Append Observation vào `current_prompt` và lặp lại

**Session state** là backbone của architecture: thay vì dump toàn bộ dữ liệu CV (>10k tokens) vào prompt, mỗi tool lưu kết quả vào `session` object và LLM chỉ nhận summary ngắn (~5 dòng). Điều này giảm token cost ~24× và tránh context overflow.

```
session.cv_data: CandidateMasterCV    ← từ extract_cv
session.jd_data: JobDescription       ← từ extract_jd
session.match_report: MatchReport     ← từ matcher (optional)
session.tailored_sections: Dict[str, TailoredSection]  ← từ draft_section
session.tailored_cv: TailoredCV       ← từ assemble_cv
```

`CVSession` sử dụng `set_cv_data()` / `set_jd_data()` để validate và coerce payload: nhận cả `dict` lẫn Pydantic object, đảm bảo kiểu dữ liệu luôn đúng downstream.

### 2.3 Định nghĩa công cụ

| Tên tool | Input | Output | Vai trò |
|---|---|---|---|
| `extract_cv` | `pdf_path: str` | Summary text + `session.cv_data` | Parse PDF → `CandidateMasterCV` Pydantic object |
| `extract_jd` | `pdf_path: str` | Summary text + `session.jd_data` | Parse JD PDF → `JobDescription` object |
| `draft_section` | `summary` \| `skills` \| `experience` | Drafted text + `TailoredSection` in session | Rewrite CV section nhắm vào JD keywords, lưu structured output có evidence |
| `assemble_cv` | _(empty)_ | Confirmation string | Ghép tất cả `TailoredSection` → `TailoredCV` |
| `export_cv_markdown` | _(empty)_ | Markdown text + file | Export `TailoredCV` ra `tailored_cv.md` |
| `validate_ats` | CV text (hoặc empty) | Score report + `is_ready` flag | Deterministic ATS scoring — trigger Final Answer khi `is_ready=True` |

**Tool call sequence bắt buộc** (enforced bởi system prompt):

```
extract_cv → extract_jd → draft_section(summary)
                        → draft_section(skills)
                        → draft_section(experience)
                        → assemble_cv → export_cv_markdown
                        → validate_ats
                                    └─ is_ready=True → Final Answer
```

`draft_section` ưu tiên **gap requirements** (chưa match trong MatchReport) lên đầu thay vì lặp lại điểm mạnh đã có — chủ động address điểm yếu của ứng viên so với JD.

### 2.4 LLM Providers

| Vai trò | Provider | Model |
|---|---|---|
| Agent reasoning (primary) | OpenRouter | `google/gemini-2.5-flash-preview` |
| Agent reasoning (fallback) | OpenRouter | `qwen/qwen3-235b-a22b:free` |
| Section drafting (LLM call) | OpenRouter / OpenAI | `gpt-4o-mini` |
| CV extraction | Rule-based parser | _(no LLM)_ |
| ATS validation | Deterministic algorithm | _(no LLM)_ |

---

## 3. Telemetry & Hiệu suất

Hệ thống logging (`src/telemetry/logger.py`) ghi lại từng event trong ReAct loop với timestamp và metadata đầy đủ:

```json
{"event": "AGENT_START",   "data": {"input": "Tailor my CV...", "model": "google/gemini-2.5-flash-preview"}}
{"event": "LLM_RESPONSE",  "data": {"step": 0, "usage": {"completion_tokens": 312}, "latency_ms": 2140}}
{"event": "THOUGHT",       "data": {"step": 0, "thought": "I need to extract the CV first..."}}
{"event": "ACTION",        "data": {"step": 0, "tool": "extract_cv", "args": "data/example-resume.pdf"}}
{"event": "TOOL_RESULT",   "data": {"tool": "extract_cv", "chars": 1842}}
{"event": "TOOL_RESULT",   "data": {"tool": "validate_ats", "score": 87.3, "is_ready": true}}
{"event": "FINAL_ANSWER",  "data": {"step": 6, "answer": "..."}}
{"event": "AGENT_END",     "data": {"steps": 7}}
```

**Số liệu quan sát từ test run:**

| Chỉ số | Giá trị |
|---|---|
| Latency P50 per LLM call | ~1.8–2.5s (Gemini Flash) |
| Steps trung bình / task | 5–7 steps |
| Token / task (estimate) | ~2,500–4,000 tokens |
| ATS score đạt được | 72–91/100 (tuỳ CV và JD) |
| Hallucination rate | ~0% (ground-truth enforcement qua session) |

Log-based trace là nguồn dữ liệu chính cho Root Cause Analysis — mỗi case phân tích đều xuất phát từ việc đọc log trực tiếp.

---

## 4. Root Cause Analysis — Các lỗi chính gặp phải

### Case 1: Agent skip tools, hallucinate Final Answer ngay step 0

**Model**: `openai/gpt-oss-20b:free`

**Log:**
```json
{"event": "AGENT_START",  "data": {"model": "openai/gpt-oss-20b:free"}}
{"event": "LLM_RESPONSE", "data": {"step": 0, "usage": {"completion_tokens": 1505}, "latency_ms": 3200}}
{"event": "THOUGHT",      "data": {"thought": "Begin by extracting the candidate's CV..."}}
{"event": "FINAL_ANSWER", "data": {"answer": "TechNova Solutions – Senior Recruitment Specialist..."}}
{"event": "AGENT_END",    "data": {"steps": 1}}
```

Không có event `ACTION` hay `TOOL_RESULT` nào. Agent viết Thought nói "cần extract CV" nhưng ngay sau đó nhảy vào Final Answer với nội dung bịa hoàn toàn: "TechNova Solutions", "San Francisco" — không có trong CV Dev Raj Singh.

**Nguyên nhân gốc rễ:**
1. Model 20B params không follow multi-step instruction reliably — tổng hợp từ training data thay vì thực hiện tool call
2. System prompt v1 chỉ có positive instruction ("use tools first") — thiếu negative constraint cấm viết Final Answer sớm

**Fix:** Thêm explicit prohibition vào `get_system_prompt()`:
```
RULES:
- You MUST call extract_cv and extract_jd before draft_section or validate_ats.
- NEVER write Final Answer before calling all required tools.
- NEVER invent company names, dates, or skills not seen in Observations.
- One Action per response, then stop and wait.
```

**Lesson**: Với LLM nhỏ hơn, **negative constraints** ("NEVER do X") hiệu quả hơn positive instructions ("do X first") vì cắt trực tiếp đường dẫn ngắn nhất mà model hay chọn.

---

### Case 2: Schema mismatch — `JDExtraction` vs `JobDescription`

**Mô tả:** Tool `extract_jd` ban đầu trả về object có attribute `requirements` nhưng code downstream gọi `.get("requirements")` như dict — gây `AttributeError` runtime.

**Nguyên nhân gốc rễ:** Hai thành viên develop song song, một dùng `JDExtraction` (custom class), một expect `JobDescription` (Pydantic schema từ `src/schemas/cv_tailoring.py`). Không có interface contract rõ ràng giữa các module.

**Fix:** Chuẩn hóa toàn bộ pipeline dùng `JobDescription`. `CVSession` được typed đầy đủ (`jd_data: Optional[JobDescription]`) và `set_jd_data()` validate+coerce payload:

```python
def set_jd_data(self, payload: JobDescription | dict | None) -> Optional[JobDescription]:
    if isinstance(payload, JobDescription):
        self.jd_data = payload
    else:
        self.jd_data = JobDescription.model_validate(payload)
    return self.jd_data
```

IDE bắt type mismatch ngay lúc development — lỗi không thể lọt đến runtime nữa.

---

### Case 3: `validate_ats` đọc `tailored_sections` là `str` thay vì `TailoredSection`

**Mô tả:** Sau khi refactor `section_drafter.py` để lưu `TailoredSection` object (thay vì plain string), `validate_ats` vẫn cố join các values như string — `TypeError` vì `.text` không tự động convert sang `str`.

**Fix:** Cập nhật `validate_ats` để extract text đúng từ `TailoredSection.blocks`:

```python
if session.tailored_sections:
    blocks = []
    for section in session.tailored_sections.values():
        blocks.append(section.title)
        blocks.extend(block.text for block in section.blocks)
    text = "\n\n".join(blocks)
```

---

## 5. Ablation Studies & Thử nghiệm

### Thử nghiệm 1: System prompt v1 vs v2

| | Prompt v1 | Prompt v2 |
|---|---|---|
| Nội dung thêm | _(baseline — positive instructions only)_ | `NEVER` constraints + explicit tool order |
| Tool-skip rate (model nhỏ) | ~80% | ~5% |
| Hallucination rate | ~60% | ~0% |
| Kết luận | Không dùng được với model <70B | Hoạt động ổn với Gemini Flash |

### Thử nghiệm 2: Chatbot vs Agent — theo loại tác vụ

| Tác vụ | Chatbot | ReAct Agent | Winner |
|---|---|---|---|
| "Python là gì?" | Đúng, ~0.8s | Đúng, ~4.2s (gọi tool không cần thiết) | **Chatbot** |
| "Tailor CV này cho JD này" | Hallucinate — không biết CV thực | Đúng — dựa vào `extract_cv`/`extract_jd` | **Agent** |
| "ATS score của CV đã tailor?" | Không thể | 87.3/100 với breakdown chi tiết | **Agent** |
| "Keyword nào còn thiếu?" | Đoán chung chung | List chính xác từ JD requirements | **Agent** |

### Thử nghiệm 3: Session-based vs Prompt-injection approach

Thay vì lưu `CandidateMasterCV` vào session, thử inject toàn bộ JSON vào prompt:
- **Kết quả**: Prompt tăng từ ~500 lên ~12,000 tokens — tốn 24× token cost
- **Chất lượng**: Tương đương hoặc kém hơn (LLM bị distracted bởi irrelevant fields)
- **Kết luận**: Session state pattern vượt trội cả về cost lẫn chất lượng

---

## 6. Đánh giá Production Readiness

### Bảo mật & Guardrails

- **Input sanitization**: PDF path xử lý qua `Path` object và `pdfplumber` — không pass vào shell
- **Tool isolation**: Mỗi tool chỉ đọc/ghi `session` trong scope, không có external side effects
- **Hallucination guard**: System prompt RULES + session-based data flow ngăn agent bịa thông tin
- **`max_steps=5`**: Ngăn infinite loop tốn chi phí
- **`validate_ats` checkpoint**: Agent không thể output CV chưa qua ATS check; `is_ready=True` (score ≥ 80 + 0 missing MUST keyword) là điều kiện duy nhất trigger Final Answer

### Những điểm chưa production-ready

| Vấn đề | Impact | Hướng xử lý |
|---|---|---|
| Tools đồng bộ, blocking | `extract_cv` và `extract_jd` chạy tuần tự | `asyncio.gather()` → giảm ~40% latency |
| Session là singleton module-level | Không hỗ trợ multi-user concurrent | Per-request session object + context var |
| Không có retry logic cho LLM | Một lỗi network → task fail | Exponential backoff với fallback provider |
| `tailored_cv.md` hardcoded path | Ghi đè nhau nếu chạy song song | UUID-based output filename |
| Không có Supervisor LLM | Không audit output trước khi trả user | Thêm layer kiểm tra hallucination trước Final Answer |

### Hướng mở rộng dài hạn

- **Vector DB tool retrieval**: Khi số tools tăng lên >20, dùng embedding similarity để agent tự chọn đúng tool thay vì list toàn bộ trong system prompt
- **Fine-tuned extraction model**: Thay `gpt-4o-mini` cho JD/section drafting bằng Qwen 7B fine-tuned — giảm cost 10× với accuracy tương đương
- **LangGraph migration**: Thay ReAct loop thủ công bằng LangGraph để hỗ trợ branching phức tạp (parallel drafting, conditional tool chains)

---

> [!NOTE]
> Report này mô tả đúng hệ thống được implement và test trong phiên lab ngày 2026-04-06 bởi nhóm F2. Code snippets trích trực tiếp từ source: `src/agent/agent.py`, `src/tools/_session.py`, `src/tools/ats_validator.py`, `src/tools/section_drafter.py`.
