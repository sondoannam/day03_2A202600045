import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> None:
        return None

load_dotenv()

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(PROJECT_ROOT))

from src.schemas import CandidateMasterCV, JobDescription
from src.tools.cv_extractor import extract_cv


CV_PATH = PROJECT_ROOT / "data" / "example-resume.pdf"
JD_PATH = PROJECT_ROOT / "data" / "jds" / "jd_recruitment_officer_final.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted"
CV_OUTPUT_PATH = OUTPUT_DIR / f"{CV_PATH.stem}.json"
JD_OUTPUT_PATH = OUTPUT_DIR / f"{JD_PATH.stem}.json"


def describe_exception(error: Exception) -> str:
    message = str(error)
    if "Incorrect API key provided" in message or "invalid_api_key" in message:
        return "JD extraction failed because OPENAI_API_KEY is invalid or still set to the placeholder value in .env."
    return message


def write_json_artifact(output_path: Path, payload: dict) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_cv_extraction() -> CandidateMasterCV:
    print("Running CV extraction stage...\n")

    extraction_result = extract_cv(str(CV_PATH))
    if extraction_result.get("error"):
        raise RuntimeError(extraction_result["error"])

    validated_cv = CandidateMasterCV.model_validate(extraction_result)
    write_json_artifact(CV_OUTPUT_PATH, validated_cv.model_dump(mode="json"))

    print("===== CV EXTRACTION COMPLETE =====")
    print(f"Output JSON: {CV_OUTPUT_PATH}")
    print(f"Candidate: {validated_cv.contact.full_name}")
    print(f"Skills: {len(validated_cv.skills)}")
    print(f"Work experiences: {len(validated_cv.work_experience)}")
    print(f"Education records: {len(validated_cv.education)}")

    if validated_cv.metadata.warnings:
        print("Warnings:")
        for warning in validated_cv.metadata.warnings:
            print(f"- {warning}")

    print()
    return validated_cv


def run_jd_extraction() -> JobDescription:
    print("Running JD extraction stage...\n")

    from src.tools.jd_extractor import extract_jd_requirements

    extracted_jd = extract_jd_requirements(str(JD_PATH))
    validated_jd = JobDescription.model_validate(extracted_jd)
    write_json_artifact(JD_OUTPUT_PATH, validated_jd.model_dump(mode="json"))

    print("===== JD EXTRACTION COMPLETE =====")
    print(f"Output JSON: {JD_OUTPUT_PATH}")
    print(f"Title: {validated_jd.title}")
    print(f"Company: {validated_jd.company_name}")
    print(f"Requirements: {len(validated_jd.requirements)}")
    print(f"Responsibilities: {len(validated_jd.responsibilities)}")
    print(f"Work arrangement: {validated_jd.work_arrangement}")

    if validated_jd.metadata.warnings:
        print("Warnings:")
        for warning in validated_jd.metadata.warnings:
            print(f"- {warning}")

    print()
    return validated_jd


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    try:
        run_cv_extraction()
    except Exception as error:
        failures.append(f"CV extraction: {describe_exception(error)}")

    try:
        run_jd_extraction()
    except Exception as error:
        failures.append(f"JD extraction: {describe_exception(error)}")

    if failures:
        print("===== EXTRACTION SUMMARY =====")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("===== EXTRACTION SUMMARY =====")
    print("- CV extraction: success")
    print("- JD extraction: success")


if __name__ == "__main__":
    main()