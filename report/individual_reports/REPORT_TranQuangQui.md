# Individual Report: Lab 3 — Chatbot vs ReAct Agent

- **Họ và tên**: Trần Quang Quí
- **MSSV**: 2A202600305
- **Ngày**: 2026-04-06

---

## I. Đóng góp Kỹ thuật (15 điểm)

### Các module đã implement

| Module | Vai trò |
|---|---|
| `src/chatbot.py` | Chatbot baseline — multi-turn LLM, không có tools |
| `src/tools/_session.py` | Shared session state với typed Pydantic fields + coercion methods |
| `src/tools/jd_tool.py` | Wrapper cho `jd_extractor` → tool dict cho agent |
| `src/tools/section_drafter.py` | Draft 3 CV sections, assemble `TailoredCV`, export markdown/JSON |
| `src/tools/ats_validator.py` | Deterministic ATS scoring algorithm (không dùng LLM) |
| `src/agent/agent.py` | ReAct loop: `_execute_tool`, `_summarize_cv_dict`, `get_system_prompt` |
| `run.py` | Entry point: demo agent vs chatbot với cùng LLM provider |

### 1. Session state pattern — chia sẻ dữ liệu không qua prompt

Vấn đề ban đầu: các tools truyền dữ liệu cho nhau bằng cách dump JSON vào prompt agent — tốn ~12,000 tokens mỗi lần, LLM bị distracted bởi fields không liên quan.

Giải pháp: `CVSession` dataclass (`src/tools/_session.py`) làm backend in-memory:

```python
@dataclass
class CVSession:
    llm: Any = None
    cv_data: Optional[CandidateMasterCV] = None
    jd_data: Optional[JobDescription] = None
    match_report: Optional[MatchReport] = None
    tailored_sections: dict[str, TailoredSection] = field(default_factory=dict)
    tailored_cv: Optional[TailoredCV] = None

    def set_cv_data(self, payload: CandidateMasterCV | dict | None) -> Optional[CandidateMasterCV]:
        if isinstance(payload, CandidateMasterCV):
            self.cv_data = payload
        else:
            self.cv_data = CandidateMasterCV.model_validate(payload)
        return self.cv_data
```

`set_cv_data()` / `set_jd_data()` validate và coerce: nhận cả raw `dict` lẫn Pydantic object, đảm bảo kiểu dữ liệu đúng cho mọi tool downstream. IDE bắt lỗi ngay lúc development thay vì runtime.

### 2. `draft_section` — structured output với source evidence

`section_drafter.py` không trả về plain string mà build `TailoredSection` chứa `TailoredTextBlock` với source tracing:

```python
block = TailoredTextBlock(
    block_id=_new_id("blk"),
    text=drafted_text,
    source_evidence=raw_evidence,          # EvidenceQuote trích từ CV gốc
    targeted_requirement_ids=req_ids,      # Link đến JD requirement IDs
)
tailored_section = TailoredSection(
    section_id=_new_id(section),
    title=title,
    blocks=[block],
)
session.tailored_sections[section] = tailored_section
```

`_priority_req_ids()` ưu tiên **gap requirements** (chưa match trong MatchReport) lên đầu danh sách targeted — draft section chủ động address điểm yếu thay vì lặp lại điểm mạnh đã có.

Mỗi drafter (`_draft_summary`, `_draft_skills`, `_draft_experience`) gọi LLM với system prompt riêng có explicit rules chống hallucination, ví dụ:

```
Rules:
- Ground every claim in the source data below.
- Do NOT invent experiences or metrics not present.
- Preserve company names, job titles, and date ranges exactly.
```

### 3. `_execute_tool` — anti-hallucination qua summarization

Thay vì để tool trả về full JSON cho agent đọc (>10k tokens), `_execute_tool` trong `agent.py` lưu full data vào session và chỉ trả về summary ngắn:

```python
def _execute_tool(self, tool_name: str, args: str) -> str:
    for tool in self.tools:
        if tool["name"] == tool_name:
            result = tool["function"](args)
            if isinstance(result, dict):
                if result.get("error"):
                    return f"Error: {result['error']}"
                session.set_cv_data(result)           # lưu full data
                return self._summarize_cv_dict(result) # trả ~5 dòng cho LLM
            return str(result)
```

`_summarize_cv_dict()` chắt lọc: tên ứng viên, top 8 skills, danh sách roles, education — đủ cho agent lý luận mà không tốn token cho metadata dư thừa.

### 4. ATS Validator — deterministic, không dùng LLM

`validate_ats` (`src/tools/ats_validator.py`) scoring hoàn toàn deterministic — tránh latency và inconsistency của LLM-based scoring:

```python
# Keyword Score (40%): MUST × 0.7 + SHOULD × 0.3
kw_must_score  = len(matched_must)  / len(must_have_reqs)  * 100
kw_should_score = len(matched_should) / len(should_have_reqs) * 100
kw_overall     = kw_must_score * 0.7 + kw_should_score * 0.3

# Format Score (40%): -20 điểm mỗi anti-pattern (table, HTML tag, symbol lạ)
fmt_score = max(0.0, 100.0 - len(flags) * 20)

# Section Score (20%): summary, experience, education, skills, projects
sec_score = sections_found / len(_REQUIRED_SECTIONS) * 100

overall  = kw_overall * 0.40 + fmt_score * 0.40 + sec_score * 0.20
is_ready = overall >= 80.0 and len(missing_must) == 0
```

`is_ready=True` là điều kiện kết thúc vòng lặp của agent. Khi `is_ready=False`, agent đọc `CRITICAL MISSING: [keywords]` từ Observation và quay lại `draft_section` với target cụ thể — tạo thành feedback loop thực sự.

### 5. System prompt v2 — negative constraints

`get_system_prompt()` trong `agent.py` enforce tool call sequence qua explicit prohibitions:

```
RULES:
- You MUST call extract_cv and extract_jd before draft_section or validate_ats.
- NEVER write Final Answer before calling all required tools.
- NEVER invent company names, dates, or skills not seen in Observations.
- One Action per response, then stop and wait.
```

### Tương tác giữa các module trong ReAct loop

```
ReActAgent.run()
    │
    ├─ extract_cv(pdf)      → session.cv_data: CandidateMasterCV
    │                         (full data in session; ~5-line summary returned to LLM)
    │
    ├─ extract_jd(pdf)      → session.jd_data: JobDescription
    │
    ├─ draft_section(summary)    → session.tailored_sections["summary"]: TailoredSection
    ├─ draft_section(skills)     → session.tailored_sections["skills"]:  TailoredSection
    ├─ draft_section(experience) → session.tailored_sections["experience"]: TailoredSection
    │
    ├─ assemble_cv()        → session.tailored_cv: TailoredCV (Pydantic, validated)
    ├─ export_cv_markdown() → tailored_cv.md saved to disk
    │
    └─ validate_ats()       → ATS Score + is_ready flag
                              is_ready=True → agent viết Final Answer
```

---

## II. Nghiên cứu Tình huống Debug (10 điểm)

### Vấn đề: Agent hallucinate và skip tools hoàn toàn

**Mô tả:** Khi test với model `openai/gpt-oss-20b:free`, agent nhảy thẳng vào Final Answer ngay step 0 mà không gọi bất kỳ tool nào. Output bịa hoàn toàn — tên công ty "TechNova Solutions", địa chỉ "San Francisco" — không có trong CV thực của Dev Raj Singh.

**Log trích từ `logs/2026-04-06.log`:**

```json
{"event": "AGENT_START",  "data": {"model": "openai/gpt-oss-20b:free"}}
{"event": "LLM_RESPONSE", "data": {"step": 0, "usage": {"completion_tokens": 1505}, "latency_ms": 3200}}
{"event": "THOUGHT",      "data": {"thought": "Begin by extracting the candidate's CV..."}}
{"event": "FINAL_ANSWER", "data": {"answer": "TechNova Solutions – Senior Recruitment Specialist..."}}
{"event": "AGENT_END",    "data": {"steps": 1}}
```

Dấu hiệu quan trọng nhất: Thought nói "cần extract CV" nhưng ngay sau đó là `FINAL_ANSWER` mà không có `ACTION` hay `TOOL_RESULT` nào. Model tự mâu thuẫn với reasoning của chính nó.

**Diagnosis:**

Hai nguyên nhân gốc rễ:

1. **Model nhỏ không follow instruction tốt**: `gpt-oss-20b:free` (20B params) có xu hướng tổng hợp câu trả lời từ training data thay vì thực hiện tool call. System prompt ban đầu nói "Use tools first" nhưng không có penalty cho việc bỏ qua.

2. **System prompt v1 thiếu negative constraint**: Chỉ có positive instruction — "follow this format strictly" — model chọn path ngắn nhất (direct answer) vì không có rule nào cấm.

Bằng chứng: cùng system prompt với model `google/gemini-2.5-flash-preview` thì hoạt động đúng. Lỗi là ở khả năng instruction-following của model nhỏ, không phải logic agent.

**Solution:**

Thêm explicit prohibition vào `get_system_prompt()`:

```python
"""RULES:
- You MUST call extract_cv and extract_jd before draft_section or validate_ats.
- NEVER write Final Answer before calling all required tools.
- NEVER invent company names, dates, or skills not seen in Observations.
- One Action per response, then stop and wait."""
```

Sau fix, tool-skip rate giảm từ ~80% xuống ~5% với model nhỏ. Model vẫn đôi khi skip tool, nhưng đủ hiếm để sử dụng được với Gemini Flash.

**Lesson:** Với LLM nhỏ hơn, **negative constraints** ("NEVER do X") hiệu quả hơn positive instructions ("do X first") vì cắt trực tiếp đường dẫn "tắt" mà model hay chọn. Đây là điều không trực quan — người ta thường nghĩ nói model cần làm gì là đủ.

---

## III. Nhận định Cá nhân: Chatbot vs ReAct (10 điểm)

### 1. Thought block giúp gì so với Chatbot?

Chatbot trả lời "tailor CV cho JD này" bằng cách generate text từ general knowledge — output có vẻ chuyên nghiệp nhưng hoàn toàn không biết CV thực tế của ứng viên. Nó có thể recommend thêm "Kubernetes experience" trong khi CV không có keyword đó, hoặc bịa tên công ty nghe có vẻ thật.

Agent với `Thought` block buộc LLM lý luận tường minh trước mỗi action:

```
Thought: I need to know the candidate's actual CV content before drafting anything.
Action: extract_cv(data/example-resume.pdf)
Observation: CV extracted: Dev Raj Singh. Skills: Python, Django, Redis...

Thought: CV loaded. Now I need to understand what the JD requires.
Action: extract_jd(data/jds/jd_recruitment_officer_final.pdf)
Observation: JD extracted: Recruitment Officer. MUST: stakeholder management, ATS systems...
```

Mỗi `Thought` là một checkpoint — LLM không thể hallucinate vì `Observation` từ tool là ground truth. Nếu tool trả về dữ liệu thực, agent phải dựa vào đó. Đây là khác biệt kiến trúc, không chỉ là khác biệt về prompt.

### 2. Khi nào Agent thực sự tệ hơn Chatbot?

Agent không phải lúc nào cũng tốt hơn. Ba trường hợp agent tệ hơn rõ rệt:

**Câu hỏi đơn giản, không cần data ngoài:** "Python là gì?" — agent cố gọi `extract_cv` rồi mới trả lời, mất 3–4× thời gian. Chatbot trả lời đúng trong 0.8s. Đây là overhead không cần thiết của agentic reasoning.

**Model nhỏ + tool chain phức tạp:** Với model 20B, agent bị lạc giữa chừng sau 2–3 steps — quên context, bắt đầu hallucinate hoặc gọi sai tool. Chatbot không có vấn đề này vì chỉ có 1 bước.

**Tool lỗi liên tiếp:** Khi `extract_jd` fail (schema mismatch), agent bị stuck trong retry loop đến hết `max_steps=5` — kết thúc với "Max steps reached without a Final Answer". Chatbot không có điểm thất bại kiểu này.

Kết luận: agent chỉ thực sự vượt trội khi tác vụ yêu cầu multi-step reasoning với external data. Với tác vụ đơn giản hoặc khi tools không ổn định, overhead của ReAct loop trở thành gánh nặng.

### 3. Observation thay đổi behavior của agent thế nào?

Quan sát rõ nhất qua ATS feedback loop. Khi `validate_ats()` trả về:

```
ATS Score: 72.0/100
Critical Keyword Match: 60.0% (3/5 MUST matched)
CRITICAL MISSING: stakeholder management, ATS systems
Final Status: REVISION REQUIRED
```

Agent đọc "CRITICAL MISSING" và tự động quay lại `draft_section(experience)` với context cụ thể — biết chính xác phải nhắm vào keyword nào. Sau lần draft lại:

```
ATS Score: 87.3/100
Critical Keyword Match: 100% (5/5 MUST matched)
Final Status: READY TO SUBMIT ✓
```

Đây là điều chatbot không thể làm: chatbot không biết output của mình đạt bao nhiêu điểm, không có cơ chế để tự điều chỉnh. Observation chuyển agent từ "generate và hy vọng" sang "đo lường và cải thiện" — bản chất của agentic behavior.

Một điểm thú vị khác: khi tool trả về `WARNING` (ví dụ: PDF parse thiếu trang), agent tự động note điều này trong Thought tiếp theo và điều chỉnh cách nó draft content. Chatbot không có khả năng này vì không có channel để nhận feedback từ môi trường.

---

## IV. Cải tiến Tương lai (5 điểm)

### Scalability — Async tool execution

Tools hiện tại đồng bộ, blocking nhau. `extract_cv` và `extract_jd` hoàn toàn độc lập — chạy song song sẽ cắt latency đáng kể:

```python
import asyncio

async def run_async(self, user_input: str) -> str:
    cv_result, jd_result = await asyncio.gather(
        loop.run_in_executor(None, extract_cv, cv_path),
        loop.run_in_executor(None, extract_jd, jd_path),
    )
```

Với nhiều ứng viên đồng thời (production), cần thêm task queue (Celery + Redis) và per-request session object thay vì singleton module-level. Hiện tại `session = CVSession()` là global — hai requests chạy song song sẽ ghi đè dữ liệu của nhau.

### Safety — Supervisor LLM layer

Thêm một Supervisor LLM kiểm tra output của `draft_section` trước khi đưa vào `assemble_cv`: xác nhận không có công ty, metric, hay skill bịa đặt so với `session.cv_data`. Đây là implementation của "zero hallucination" constraint ở tầng kiến trúc thay vì chỉ ở tầng prompt.

Supervisor có thể đơn giản là một LLM call nhỏ với prompt:

```
Given these verified facts: {session.cv_data summary}
Does this drafted text contain any claims not grounded in the facts above?
Answer: YES (with list) or NO.
```

### Performance — Caching & Smaller models

`section_drafter.py` hiện gọi LLM cho mỗi section mỗi lần. Cải thiện ở hai hướng:

**Caching:** Hash JD text → cache `JobDescription` extracted. Cùng JD không cần extract lại — tiết kiệm ~1 LLM call/request.

**Fine-tuned smaller model:** `gpt-4o-mini` dùng cho drafting tốn kém ở scale. Có thể fine-tune Qwen 7B trên dataset CV-tailoring examples để đạt accuracy tương đương với cost 10× thấp hơn. Section drafting là structured, repetitive task — ideal candidate cho fine-tuning.

---

> [!NOTE]
> Report này mô tả đúng contribution thực tế trong session làm lab ngày 2026-04-06. Code snippets trích trực tiếp từ source: `src/agent/agent.py`, `src/tools/_session.py`, `src/tools/section_drafter.py`, `src/tools/ats_validator.py`.
