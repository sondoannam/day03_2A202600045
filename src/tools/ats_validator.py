import re
from src.telemetry.logger import logger
from src.schemas.cv_tailoring import *

# Định nghĩa các Header chuẩn để quét Section Completeness (Theo     Section 5)
_REQUIRED_SECTIONS = ["summary", "experience", "education", "skills", "projects"]

_FORMAT_ANTI_PATTERNS = [
    (r"\|.+\|.+\|", "Table detected — ATS may fail to parse columns"),
    (r"<[a-zA-Z]+>", "HTML tags detected — strip before submission"),
    (r"[\u2022\u2023\u25B6\u25C0\u25A0]", "Complex symbols detected — use standard bullets (- or *)"),
]

def validate_ats(args: str) -> str:
    from src.tools._session import session

    text = args.strip()
    if not text:
        if session.tailored_cv:
            blocks = []
            for section in session.tailored_cv.sections:
                blocks.append(section.title)
                blocks.extend(block.text for block in section.blocks)
            text = "\n\n".join(blocks)
        elif session.tailored_sections:
            blocks = []
            for section in session.tailored_sections.values():
                blocks.append(section.title)
                blocks.extend(block.text for block in section.blocks)
            text = "\n\n".join(blocks)
        else:
            return "ERROR: No CV text to validate. Provide text or call draft_section first."

    if not session.jd_data:
        return "ERROR: No JD loaded. Call extract_jd first."

    # Lấy dữ liệu từ Schema mới
    jd: JobDescription = session.jd_data
    cv_lower = text.lower()

    # 1. Phân loại Keywords từ JobDescription.requirements
    must_have_reqs = [r for r in jd.requirements if r.priority == RequirementPriority.MUST]
    should_have_reqs = [r for r in jd.requirements if r.priority == RequirementPriority.SHOULD]
    nice_to_have_reqs = [r for r in jd.requirements if r.priority == RequirementPriority.NICE_TO_HAVE]

    # MVP Check: Kiểm tra sự tồn tại (Deterministic Match)
    matched_must = [r.text for r in must_have_reqs if r.text.lower() in cv_lower]
    missing_must = [r.text for r in must_have_reqs if r.text.lower() not in cv_lower]
    
    matched_should = [r.text for r in should_have_reqs if r.text.lower() in cv_lower]
    matched_nice = [r.text for r in nice_to_have_reqs if r.text.lower() in cv_lower]

    # 2. Tính toán điểm số theo trọng số Blueprint (Section 5.4)
    # Keyword Score (40%): Tập trung vào MUST và SHOULD
    must_weight = 0.7
    should_weight = 0.3
    
    kw_must_score = (len(matched_must) / len(must_have_reqs) * 100) if must_have_reqs else 100.0
    kw_should_score = (len(matched_should) / len(should_have_reqs) * 100) if should_have_reqs else 100.0
    kw_overall_score = (kw_must_score * must_weight) + (kw_should_score * should_weight)

    # Section Score (20%): Kiểm tra các tiêu đề mục
    sections_found = sum(1 for s in _REQUIRED_SECTIONS if s in cv_lower)
    sec_score = (sections_found / len(_REQUIRED_SECTIONS)) * 100

    # Format Score (40%): Trừ điểm nếu vi phạm Anti-patterns
    flags = [msg for pat, msg in _FORMAT_ANTI_PATTERNS if re.search(pat, text)]
    fmt_score = max(0.0, 100.0 - (len(flags) * 20))

    # Overall Weighted Score
    overall = (kw_overall_score * 0.40) + (fmt_score * 0.40) + (sec_score * 0.20)

    # 3. Điều kiện Dừng (Termination Logic) cho Agent
    # Is Ready chỉ TRUE khi điểm cao VÀ không thiếu bất kỳ từ khóa MUST nào
    is_ready = overall >= 80.0 and len(missing_must) == 0

    logger.log_event("TOOL_RESULT", {
        "tool": "validate_ats",
        "score": round(overall, 1),
        "missing_must": missing_must,
        "is_ready": is_ready,
    })

    # 4. Format Output cho Agent đọc (Observation)
    lines = [
        f"--- ATS COMPLIANCE REPORT ---",
        f"Overall Match Score: {overall:.1f}/100",
        f"  - Critical Keyword Match: {kw_must_score:.1f}% ({len(matched_must)}/{len(must_have_reqs)} MUST matched)",
        f"  - Contextual Match: {kw_should_score:.1f}% ({len(matched_should)}/{len(should_have_reqs)} SHOULD matched)",
        f"  - Format Integrity: {fmt_score:.1f}%",
        f"  - Section Completeness: {sec_score:.1f}%",
    ]

    if missing_must:
        lines.append(f"CRITICAL MISSING: {', '.join(missing_must)}")
    
    if flags:
        lines.append(f"FORMAT ISSUES: {'; '.join(flags)}")

    if matched_nice:
        lines.append(f"BONUS POINTS: Matched {len(matched_nice)} nice-to-have items.")

    status = "READY TO SUBMIT ✓" if is_ready else "REVISION REQUIRED: Ensure all MUST keywords are included and score >= 80."
    lines.append(f"\nFinal Status: {status}")

    return "\n".join(lines)

ats_validator_tool = {
    "name": "validate_ats",
    "description": (
        "Performs a local, deterministic ATS scan of the tailored CV. "
        "Evaluates mandatory keyword coverage (MUST), format anti-patterns, and section presence. "
        "This is NOT an LLM call. It uses the previously extracted JD data. "
        "The agent must fix all missing 'MUST' keywords before providing a Final Answer."
    ),
    "function": validate_ats,
}