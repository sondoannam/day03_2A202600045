import os
import json
from typing import Callable, List, Tuple
from openai import OpenAI
import instructor
from dotenv import load_dotenv

load_dotenv()

from src.schemas.cv_tailoring import MatchReport
from src.tools._session import session

def _is_configured_key(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered not in {
        "",
        "your_openai_api_key_here",
        "your_openrouter_api_key_here",
    }

def _provider_attempts() -> List[Tuple[str, str, Callable[[], OpenAI]]]:
    attempts: List[Tuple[str, str, Callable[[], OpenAI]]] = []
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if _is_configured_key(openrouter_key):
        attempts.append(
            (
                "openrouter",
                os.getenv("OPENROUTER_JD_MODEL", "openrouter/free"),
                lambda: OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                ),
            )
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if _is_configured_key(openai_key):
        attempts.append(
            (
                "openai",
                os.getenv("OPENAI_JD_MODEL", "gpt-4o-mini"),
                lambda: OpenAI(api_key=openai_key),
            )
        )

    return attempts

def execute_matching_llm(jd_json: str, cv_json: str) -> MatchReport:
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "docs", "prompt", "tool02", "MASTER_PROMPT.md"
    )
    master_prompt = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            master_prompt = f.read()

    # Nhấn mạnh với mô hình phải ưu tiên schema được Inject từ Instructor / Pydantic (tức class MatchReport của codebase) thay vì schema mẫu trong file md.
    sys_prompt = master_prompt + "\n\nQUAN TRỌNG: Bạn BẮT BUỘC bỏ qua ví dụ Schema ở trên và CHỈ XUẤT JSON trả về tương ứng trinh khớp hoàn toàn với FUNCTION STRUCTURE / PYDANTIC SCHEMA được truyền vào theo system definition."
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": f"[JOB_DESCRIPTION]\n{jd_json}\n\n[CANDIDATE_CV]\n{cv_json}\n\nHãy xuất kết quả JSON với schema MatchReport."
        }
    ]

    attempts = _provider_attempts()
    if not attempts:
        raise RuntimeError("No LLM provider is configured. Please set OPENROUTER_API_KEY or OPENAI_API_KEY in .env.")

    provider_errors: List[str] = []

    for provider_name, model_name, client_factory in attempts:
        try:
            client = instructor.from_openai(client_factory())
            response_data = client.chat.completions.create(
                model=model_name,
                response_model=MatchReport,
                messages=messages,
                max_retries=2,
            )
            return response_data
        except Exception as e:
            provider_errors.append(f"{provider_name} ({model_name}): {e}")

    raise RuntimeError("All MatchReport matching LLM providers failed. " + " | ".join(provider_errors))

def match_cv_jd(args: str = "") -> str:
    """Tool function to execute AI Matching against stored session data."""
    if not session.cv_data:
        return "ERROR: Missing CV data in session. Please run cv_extractor tool (Tool 01) first."
    if not session.jd_data:
        return "ERROR: Missing JD data in session. Please run jd_extractor tool first."
        
    cv_json = session.cv_data.model_dump_json(exclude_none=True)
    jd_json = session.jd_data.model_dump_json(exclude_none=True)
    
    try:
        report = execute_matching_llm(jd_json, cv_json)
        session.match_report = report
        return f"Match completed successfully! Overall Score: {report.overall_score}%. Báo cáo MatchReport đã được lưu vào hệ thống nội bộ để sử dụng cho task tiếp theo."
    except Exception as e:
        return f"ERROR during matching process: {str(e)}"

cv_jd_matcher_tool = {
    "name": "match_cv_jd",
    "description": (
        "So sánh JobDescription và CandidateMasterCV để xuất ra báo cáo MatchReport chi tiết. "
        "Chạy tool này SAU KHI CV và JD đã được extract vào session variables qua cv_extractor và jd_extractor. "
        "Input (args) không bắt buộc. "
    ),
    "function": match_cv_jd,
}
