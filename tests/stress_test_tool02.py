import os
import json
import sys
from pydantic import BaseModel
from typing import List, Optional, Literal
from openai import OpenAI
import instructor
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools.cv_extractor import extract_cv

load_dotenv()

class RequirementMatch(BaseModel):
    requirement_id: str
    requirement_description: str
    priority: Literal["MANDATORY", "PREFERRED"]
    matched: bool
    match_type: Literal["EXACT", "SEMANTIC", "PARTIAL", "NO_MATCH"]
    score: float
    cv_evidence: str
    gap_analysis: Optional[str]

class MatchReport(BaseModel):
    keyword_score: float
    overall_match_rate: float
    is_qualified: bool
    matches: List[RequirementMatch]

class Tool02Output(BaseModel):
    match_report: MatchReport

def run_stress_test():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.")
        return

    client = instructor.from_openai(
        OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    )
    
    prompt_path = "docs/prompt/tool02/MASTER_PROMPT.md"
    if not os.path.exists(prompt_path):
        print(f"File {prompt_path} not found.")
        return
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        master_prompt = f.read()
        
    cv_path = "data/Dev-Raj-Resume.pdf"
    print(f"Đang trích xuất nội dung CV từ {cv_path}...")
    candidate_cv_dict = extract_cv(cv_path)
    sample_cv = json.dumps(candidate_cv_dict, ensure_ascii=False, indent=2)
    
    sample_jd = """
    Job Title: Recruitment Officer
    Requirements:
    - REQ_001: Có kinh nghiệm viết C++ và Golang (MANDATORY).
    - REQ_002: Có chứng chỉ AWS Certified (PREFERRED).
    """

    messages = [
        {
            "role": "system",
            "content": master_prompt
        },
        {
            "role": "user",
            "content": f"[JOB_DESCRIPTION]\n{sample_jd}\n\n[CANDIDATE_CV]\n{sample_cv}\n\nHãy xuất kết quả đối soát."
        }
    ]

    print("Running LLM request to evaluate prompt deterministic parsing qua OpenRouter Free...")
    
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            response_model=Tool02Output,
            messages=messages,
        )
        print("✅ SUCCESS! Parsed result:")
        print(response.model_dump_json(indent=2))
    except Exception as e:
        print("❌ FAILED!")
        print(e)

if __name__ == "__main__":
    run_stress_test()
