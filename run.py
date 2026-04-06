import argparse
import asyncio
import json
import os
import sys
from importlib import import_module
from pathlib import Path

def load_dotenv() -> None:
    try:
        import_module("dotenv").load_dotenv()
    except ImportError:
        return None


load_dotenv()

PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(PROJECT_ROOT))

from src.core.provider_factory import create_provider
from src.schemas import CandidateMasterCV, JobDescription, MatchReport, SourceType
from src.tools._session import session
from src.tools.ats_validator import validate_ats
from src.tools.cv_extractor import extract_cv
from src.tools.cv_jd_matcher import match_cv_jd
from src.tools.jd_extractor import extract_jd_requirements, extract_jd_text_requirements
from src.tools.section_drafter import export_cv_markdown, generate_cv_json
from src.tools.JD_Web_Scraper import JD_Web_Scraper


CV_PATH = PROJECT_ROOT / "data" / "example-resume.pdf"
JD_PATH = PROJECT_ROOT / "data" / "jds" / "jd_recruitment_officer_final.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CV Tailoring System end-to-end pipeline.")
    parser.add_argument("--cv", default=str(CV_PATH), help="Path to the master CV PDF.")
    parser.add_argument("--jd-pdf", default=None, help="Path to the JD PDF file.")
    parser.add_argument("--jd-url", default=None, help="Job description URL to scrape and extract.")
    parser.add_argument("--provider", default=None, help="LLM provider override. Defaults to DEFAULT_PROVIDER (Gemini).")
    parser.add_argument("--model", default=None, help="Model override for the selected provider.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to write pipeline artifacts.")
    return parser.parse_args()


def describe_exception(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()

    if "invalid_api_key" in lowered or "incorrect api key" in lowered:
        return "A provider API key is invalid or still set to the placeholder value in .env."
    if "api key" in lowered and "configured" in lowered:
        return message
    return message


def write_json_artifact(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_artifact(output_path: Path, payload: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")


def run_cv_extraction(cv_path: Path, output_dir: Path) -> CandidateMasterCV:
    print("Running CV extraction stage...\n")

    extraction_result = extract_cv(str(cv_path))
    if extraction_result.get("error"):
        raise RuntimeError(extraction_result["error"])

    validated_cv = CandidateMasterCV.model_validate(extraction_result)
    write_json_artifact(output_dir / f"{cv_path.stem}.json", validated_cv.model_dump(mode="json"))
    session.set_cv_data(validated_cv)

    print("===== CV EXTRACTION COMPLETE =====")
    print(f"Output JSON: {output_dir / f'{cv_path.stem}.json'}")
    print(f"Candidate: {validated_cv.contact.full_name}")
    print(f"Skills: {len(validated_cv.skills)}")
    print(f"Work experiences: {len(validated_cv.work_experience)}")
    print(f"Education records: {len(validated_cv.education)}")
    print()
    return validated_cv


def run_jd_extraction_from_pdf(jd_pdf_path: Path, output_dir: Path) -> JobDescription:
    print("Running JD extraction stage (PDF)...\n")

    extracted_jd = extract_jd_requirements(str(jd_pdf_path))
    validated_jd = JobDescription.model_validate(extracted_jd)
    write_json_artifact(output_dir / f"{jd_pdf_path.stem}.json", validated_jd.model_dump(mode="json"))
    session.set_jd_data(validated_jd)

    print("===== JD EXTRACTION COMPLETE =====")
    print(f"Output JSON: {output_dir / f'{jd_pdf_path.stem}.json'}")
    print(f"Title: {validated_jd.title}")
    print(f"Company: {validated_jd.company_name}")
    print(f"Requirements: {len(validated_jd.requirements)}")
    print(f"Responsibilities: {len(validated_jd.responsibilities)}")
    print()
    return validated_jd


def run_jd_extraction_from_url(jd_url: str, output_dir: Path) -> JobDescription:
    print("Running JD extraction stage (URL)...\n")

    scraper = JD_Web_Scraper()
    scrape_result = json.loads(asyncio.run(scraper.execute(jd_url)))
    raw_markdown = scrape_result.get("raw_markdown", "").strip()
    if not raw_markdown:
        raise RuntimeError("JD URL scraping did not return usable text content.")

    extracted_jd = extract_jd_text_requirements(raw_markdown)
    validated_jd = JobDescription.model_validate(extracted_jd)
    validated_jd.metadata.source_type = SourceType.URL
    validated_jd.metadata.source_uri = jd_url

    write_json_artifact(output_dir / "jd_from_url.json", validated_jd.model_dump(mode="json"))
    write_text_artifact(output_dir / "jd_from_url_raw.md", raw_markdown)
    session.set_jd_data(validated_jd)

    print("===== JD EXTRACTION COMPLETE =====")
    print(f"Output JSON: {output_dir / 'jd_from_url.json'}")
    print(f"Raw markdown: {output_dir / 'jd_from_url_raw.md'}")
    print(f"Title: {validated_jd.title}")
    print(f"Company: {validated_jd.company_name}")
    print(f"Requirements: {len(validated_jd.requirements)}")
    print(f"Responsibilities: {len(validated_jd.responsibilities)}")
    print()
    return validated_jd


def run_matching(output_dir: Path) -> MatchReport:
    print("Running CV/JD matching stage...\n")

    match_result = match_cv_jd("")
    if match_result.startswith("ERROR"):
        raise RuntimeError(match_result)
    if not session.match_report:
        raise RuntimeError("Matcher completed without storing MatchReport in session.")

    write_json_artifact(output_dir / "match_report.json", session.match_report.model_dump(mode="json"))

    print("===== MATCHING COMPLETE =====")
    print(match_result)
    print(f"Output JSON: {output_dir / 'match_report.json'}")
    print()
    return session.match_report


def run_tailored_cv_generation(output_dir: Path) -> Path:
    print("Running tailored CV generation stage...\n")

    output_path = output_dir / "tailored_cv.json"
    generation_result = generate_cv_json(str(output_path))
    if generation_result.startswith("ERROR"):
        raise RuntimeError(generation_result)

    markdown_result = export_cv_markdown("")
    if markdown_result.startswith("ERROR"):
        raise RuntimeError(markdown_result)

    markdown_path = PROJECT_ROOT / "tailored_cv.md"
    if markdown_path.exists():
        write_text_artifact(output_dir / "tailored_cv.md", markdown_path.read_text(encoding="utf-8"))

    print("===== TAILORED CV COMPLETE =====")
    print(generation_result)
    print(f"Markdown CV: {output_dir / 'tailored_cv.md'}")
    print()
    return output_path


def run_ats_validation(output_dir: Path) -> str:
    print("Running ATS validation stage...\n")

    report = validate_ats("")
    if report.startswith("ERROR"):
        raise RuntimeError(report)

    write_text_artifact(output_dir / "ats_validation.txt", report)

    print("===== ATS VALIDATION COMPLETE =====")
    print(report)
    print()
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cv_path = Path(args.cv).expanduser().resolve()
    jd_pdf_path = Path(args.jd_pdf).expanduser().resolve() if args.jd_pdf else JD_PATH.resolve()

    session.reset()
    session.llm = create_provider(args.provider, args.model)

    print("===== PIPELINE CONFIG =====")
    print(f"Provider: {session.llm.__class__.__name__}")
    print(f"Model: {session.llm.model_name}")
    print(f"CV source: {cv_path}")
    print(f"JD source: {args.jd_url if args.jd_url else jd_pdf_path}")
    print(f"Output dir: {output_dir}")
    print()

    failures: list[str] = []

    try:
        run_cv_extraction(cv_path, output_dir)
    except Exception as error:
        failures.append(f"CV extraction: {describe_exception(error)}")

    if not failures:
        try:
            if args.jd_url:
                run_jd_extraction_from_url(args.jd_url, output_dir)
            else:
                run_jd_extraction_from_pdf(jd_pdf_path, output_dir)
        except Exception as error:
            failures.append(f"JD extraction: {describe_exception(error)}")

    if not failures:
        try:
            run_matching(output_dir)
        except Exception as error:
            failures.append(f"Matching: {describe_exception(error)}")

    if not failures:
        try:
            run_tailored_cv_generation(output_dir)
        except Exception as error:
            failures.append(f"Tailored CV generation: {describe_exception(error)}")

    if not failures:
        try:
            run_ats_validation(output_dir)
        except Exception as error:
            failures.append(f"ATS validation: {describe_exception(error)}")

    if failures:
        print("===== PIPELINE SUMMARY =====")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("===== PIPELINE SUMMARY =====")
    print("- CV extraction: success")
    print("- JD extraction: success")
    print("- Matching: success")
    print("- Tailored CV generation: success")
    print("- ATS validation: success")


if __name__ == "__main__":
    main()