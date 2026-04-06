import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools._session import session
from src.tools.cv_extractor import extract_cv
from src.tools.cv_jd_matcher import match_cv_jd

def test_matcher():
    # 1. Prepare JD Data
    print("Mocking JD Data...")
    sample_jd_dict = {
        "metadata": {
            "source_type": "text",
            "extractor_name": "mock",
            "schema_version": "1.0",
            "language": "vi"
        },
        "title": "Recruitment Officer",
        "requirements": [
            {
                "requirement_id": "REQ_001",
                "category": "hard_skill",
                "text": "Có kinh nghiệm viết C++ và Golang",
                "priority": "must",
                "required": True
            },
            {
                "requirement_id": "REQ_002",
                "category": "certification",
                "text": "Có chứng chỉ AWS Certified Cloud Practitioner",
                "priority": "nice_to_have",
                "required": False
            }
        ]
    }
    try:
        session.set_jd_data(sample_jd_dict)
    except Exception as e:
        print("JD Validation Error:", e)
        return
    
    # 2. Prepare CV Data
    cv_path = "data/Dev-Raj-Resume.pdf"
    if not os.path.exists(cv_path):
        print(f"Lỗi: Không tìm thấy file {cv_path}")
        return
        
    print(f"Trích xuất CV từ {cv_path}...")
    candidate_cv_dict = extract_cv(cv_path)
    try:
        session.set_cv_data(candidate_cv_dict)
    except Exception as e:
        print("CV Validation Error:", e)
        return
    
    # 3. Test the Tool
    print("Executing match_cv_jd tool...")
    result_msg = match_cv_jd()
    
    print("\n[TOOL OUTPUT MESSAGE]")
    print(result_msg)
    
    print("\n[MATCH REPORT IN SESSION STATE]")
    if session.match_report:
        print(session.match_report.model_dump_json(indent=2))
    else:
        print("None. Trích xuất thất bại, session.match_report vẫn là None.")

if __name__ == "__main__":
    test_matcher()
