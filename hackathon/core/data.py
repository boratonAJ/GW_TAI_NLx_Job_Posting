from pathlib import Path
import zipfile

import pandas as pd

from hackathon.core.nlp_pipeline import (
    build_skill_profiles_from_mentions,
    extract_skill_mentions_from_text,
    infer_education_and_experience,
)


REQUIRED_JOB_COLUMNS = [
    "system_job_id",
    "title",
    "description",
    "city",
    "zipcode",
    "parameters_salary_min",
    "parameters_salary_max",
    "requirements_min_education",
    "requirements_experience",
    "classifications_onet_code",
    "moc_codes",
    "cip_codes",
    "application_company",
    "link",
    "created_date",
    "last_updated_date",
    "parameters_salary_unit",
    "parameters_positions_max",
    "requirements_license",
    "requirements_training",
    "rapids_codes",
    "ghostjob",
    "fedcontractor",
    "jobclass",
    "expired",
    "expired_date",
    "date_compiled",
    "source_state",
    "classifications_naics_code",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def raw_data_dir() -> Path:
    return project_root() / "data" / "raw"


def processed_data_dir() -> Path:
    return project_root() / "data" / "processed"


def selected_zip_dir() -> Path:
    return project_root() / "data" / "Colorado-Hackathon-Dataset-selected"


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def prepare_raw_data() -> tuple[Path, Path]:
    output_dir = raw_data_dir()
    jobs_csv = output_dir / "colorado.csv"
    processed_csv = output_dir / "colorado_processed.csv"

    if jobs_csv.exists() and processed_csv.exists():
        return jobs_csv, processed_csv

    zip_dir = selected_zip_dir()
    jobs_zip = zip_dir / "colorado.csv.zip"
    processed_zip = zip_dir / "colorado_processed.csv.zip"

    if not jobs_zip.exists() or not processed_zip.exists():
        raise FileNotFoundError(
            "Could not find colorado zip files. Expected at "
            f"{jobs_zip} and {processed_zip}"
        )

    _extract_zip(jobs_zip, output_dir)
    _extract_zip(processed_zip, output_dir)

    if not jobs_csv.exists() or not processed_csv.exists():
        raise FileNotFoundError(
            "Zip extraction completed but CSV files were not found in data/raw."
        )

    return jobs_csv, processed_csv


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _nlp_artifact_paths() -> tuple[Path, Path]:
    output_dir = processed_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    mentions_path = output_dir / "nlp_skill_mentions.csv"
    profiles_path = output_dir / "nlp_skill_profiles.csv"
    return mentions_path, profiles_path


def _requirements_profile_path() -> Path:
    output_dir = processed_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "nlp_requirements_profile.csv"


def _generate_requirements_profile(jobs_clean: pd.DataFrame) -> pd.DataFrame:
    requirements_profile = infer_education_and_experience(jobs_clean)
    requirements_profile = _ensure_columns(
        requirements_profile,
        [
            "system_job_id",
            "education_display",
            "education_source",
            "experience_display",
            "experience_source",
        ],
    ).fillna("")
    requirements_profile["system_job_id"] = requirements_profile["system_job_id"].astype(str)
    requirements_profile.to_csv(_requirements_profile_path(), index=False)
    return requirements_profile


def _load_cached_requirements_profile() -> pd.DataFrame | None:
    requirements_path = _requirements_profile_path()
    if not requirements_path.exists():
        return None

    requirements_profile = pd.read_csv(requirements_path)
    requirements_profile = _ensure_columns(
        requirements_profile,
        [
            "system_job_id",
            "education_display",
            "education_source",
            "experience_display",
            "experience_source",
        ],
    ).fillna("")

    if requirements_profile.empty:
        return None

    requirements_profile["system_job_id"] = requirements_profile["system_job_id"].astype(str)
    requirements_profile = requirements_profile.drop_duplicates(subset=["system_job_id"], keep="first")
    return requirements_profile


def _generate_nlp_structured_data(
    jobs_clean: pd.DataFrame,
    processed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mentions = extract_skill_mentions_from_text(jobs_clean, processed)

    if mentions.empty:
        fallback_mentions = processed[["Research ID", "Taxonomy Skill"]].copy()
        fallback_mentions["NLP Score"] = processed["Correlation Coefficient"]
        mentions = fallback_mentions

    mentions["Research ID"] = mentions["Research ID"].astype(str)
    mentions["Taxonomy Skill"] = mentions["Taxonomy Skill"].astype(str)

    skill_profiles = build_skill_profiles_from_mentions(mentions)
    skill_profiles["system_job_id"] = skill_profiles["system_job_id"].astype(str)

    mentions_path, profiles_path = _nlp_artifact_paths()
    mentions.to_csv(mentions_path, index=False)
    skill_profiles.to_csv(profiles_path, index=False)

    return mentions, skill_profiles


def _load_cached_nlp_structured_data() -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    mentions_path, profiles_path = _nlp_artifact_paths()
    if not mentions_path.exists() or not profiles_path.exists():
        return None, None

    mentions = pd.read_csv(mentions_path)
    skill_profiles = pd.read_csv(profiles_path)

    mentions = _ensure_columns(mentions, ["Research ID", "Taxonomy Skill", "NLP Score"])
    skill_profiles = _ensure_columns(skill_profiles, ["system_job_id", "skill_text"])

    if mentions.empty or skill_profiles.empty:
        return None, None

    mentions["Research ID"] = mentions["Research ID"].astype(str)
    skill_profiles["system_job_id"] = skill_profiles["system_job_id"].astype(str)

    return mentions, skill_profiles


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jobs_path, processed_path = prepare_raw_data()

    jobs = pd.read_csv(jobs_path)
    processed = pd.read_csv(processed_path)

    jobs = _ensure_columns(jobs, REQUIRED_JOB_COLUMNS)
    jobs_clean = jobs[REQUIRED_JOB_COLUMNS].fillna("").copy()
    jobs_clean["system_job_id"] = jobs_clean["system_job_id"].astype(str)

    requirements_profile = _load_cached_requirements_profile()
    if requirements_profile is None:
        requirements_profile = _generate_requirements_profile(jobs_clean)
    else:
        required_ids = set(jobs_clean["system_job_id"])
        cached_ids = set(requirements_profile["system_job_id"])
        if required_ids != cached_ids:
            requirements_profile = _generate_requirements_profile(jobs_clean)

    jobs_clean = jobs_clean.merge(requirements_profile, on="system_job_id", how="left")
    jobs_clean = jobs_clean.fillna("")

    required_processed_columns = ["Research ID", "Taxonomy Skill", "Correlation Coefficient"]
    processed = _ensure_columns(processed, required_processed_columns).fillna("")
    processed["Research ID"] = processed["Research ID"].astype(str)

    nlp_mentions, skill_profiles = _load_cached_nlp_structured_data()
    if nlp_mentions is None or skill_profiles is None:
        nlp_mentions, skill_profiles = _generate_nlp_structured_data(jobs_clean, processed)

    return jobs_clean, skill_profiles, nlp_mentions


def load_raw_skill_records() -> pd.DataFrame:
    _, processed_path = prepare_raw_data()
    processed = pd.read_csv(processed_path)
    expected_columns = [
        "Research ID",
        "Raw Skill",
        "Taxonomy Skill",
        "Taxonomy Source",
        "Correlation Coefficient",
    ]
    processed = _ensure_columns(processed, expected_columns).fillna("")
    processed["Research ID"] = processed["Research ID"].astype(str)
    processed["Correlation Coefficient"] = pd.to_numeric(
        processed["Correlation Coefficient"], errors="coerce"
    ).fillna(0)
    return processed


def prepare_nlp_artifacts() -> tuple[Path, Path, Path]:
    load_data()
    mentions_path, profiles_path = _nlp_artifact_paths()
    return mentions_path, profiles_path, _requirements_profile_path()
