from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def format_salary(salary_min, salary_max, salary_unit: str = "") -> str:
    try:
        minimum = float(salary_min) if str(salary_min).strip() not in {"", "nan", "0", "0.0"} else 0
        maximum = float(salary_max) if str(salary_max).strip() not in {"", "nan", "0", "0.0"} else 0
        if minimum <= 0:
            return "Not disclosed"

        suffix = ""
        unit = str(salary_unit).strip()
        if unit and unit.lower() != "nan":
            suffix = f" / {unit}"
        elif maximum > 0:
            suffix = " / year"

        if maximum > 0:
            return f"${minimum:,.0f} — ${maximum:,.0f}{suffix}"
        return f"${minimum:,.0f}+{suffix}"
    except (TypeError, ValueError):
        return "Not disclosed"


def detect_emerging_skills(
    processed_raw: pd.DataFrame,
    confidence_threshold: float = 0.65,
    min_employers: int = 2,
    top_n: int = 30,
) -> pd.DataFrame:
    required = {"Research ID", "Raw Skill", "Taxonomy Skill", "Taxonomy Source", "Correlation Coefficient"}
    if processed_raw.empty or not required.issubset(processed_raw.columns):
        return pd.DataFrame()

    uncertain = processed_raw[
        (pd.to_numeric(processed_raw["Correlation Coefficient"], errors="coerce").fillna(0) < confidence_threshold)
        & (processed_raw["Raw Skill"].astype(str).str.len() > 3)
    ].copy()
    if uncertain.empty:
        return pd.DataFrame()

    grouped = uncertain.groupby("Raw Skill", as_index=False).agg(
        employer_count=("Research ID", "nunique"),
        avg_confidence=("Correlation Coefficient", "mean"),
        closest_taxonomy=("Taxonomy Skill", lambda values: values.mode().iloc[0] if len(values.mode()) else ""),
        taxonomy_source=("Taxonomy Source", lambda values: values.mode().iloc[0] if len(values.mode()) else ""),
    )

    grouped = grouped[grouped["employer_count"] >= min_employers]
    grouped = grouped[~grouped["Raw Skill"].astype(str).str.match(r"^\d+$")]
    grouped = grouped[grouped["Raw Skill"].astype(str).str.strip().str.len() > 4]

    return grouped.sort_values("employer_count", ascending=False).head(top_n)


def analyze_ghost_job_language(jobs_clean: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if jobs_clean.empty or "description" not in jobs_clean.columns:
        return None, None

    ghost_jobs = jobs_clean[(jobs_clean.get("is_ghost", False)) & (jobs_clean["description"].astype(str).str.len() > 20)]
    real_jobs = jobs_clean[(~jobs_clean.get("is_ghost", False)) & (jobs_clean["description"].astype(str).str.len() > 20)]
    if len(ghost_jobs) < 5 or len(real_jobs) < 5:
        return None, None

    ghost_text = " ".join(ghost_jobs["description"].astype(str).tolist())
    real_text = " ".join(real_jobs["description"].astype(str).tolist())
    if not ghost_text.strip() or not real_text.strip():
        return None, None

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    vectorizer.fit([ghost_text, real_text])

    ghost_vector = vectorizer.transform([ghost_text]).toarray()[0]
    real_vector = vectorizer.transform([real_text]).toarray()[0]

    comparison = pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out(),
            "ghost_tfidf": ghost_vector,
            "real_tfidf": real_vector,
        }
    )
    comparison["ghost_dominance"] = comparison["ghost_tfidf"] - comparison["real_tfidf"]

    ghost_terms = comparison.nlargest(15, "ghost_dominance")[["term", "ghost_tfidf"]].rename(
        columns={"term": "Term", "ghost_tfidf": "TF-IDF Score"}
    )
    real_terms = comparison.nlargest(15, "real_tfidf")[["term", "real_tfidf"]].rename(
        columns={"term": "Term", "real_tfidf": "TF-IDF Score"}
    )
    return ghost_terms, real_terms


def detect_credential_inflation(jobs_clean: pd.DataFrame) -> pd.DataFrame:
    education_rank = {
        "No Formal Education Required": 0,
        "High School Diploma Or Ged": 1,
        "Some College Coursework Completed": 2,
        "Associate'S Degree": 3,
        "Associates Degree": 3,
        "Associate Degree": 3,
        "Bachelor'S Degree": 4,
        "Bachelors Degree": 4,
        "Bachelor Degree": 4,
        "Master'S Degree": 5,
        "Masters Degree": 5,
        "Master Degree": 5,
        "Doctoral Degree": 6,
        "Phd": 6,
        "Post-Doctoral Training": 7,
        "Post Doctoral Training": 7,
    }
    inverse_rank = {
        0: "No Formal Education Required",
        1: "High School Diploma or GED",
        2: "Some College Coursework Completed",
        3: "Associate's Degree",
        4: "Bachelor's Degree",
        5: "Master's Degree",
        6: "Doctoral Degree",
        7: "Post-Doctoral Training",
    }

    required_columns = {"requirements_min_education", "classifications_onet_code"}
    if jobs_clean.empty or not required_columns.issubset(jobs_clean.columns):
        return pd.DataFrame()

    working = jobs_clean.copy()
    working["edu_normalized"] = (
        working["requirements_min_education"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    eligible = working[
        working["edu_normalized"].isin(education_rank.keys())
        & (working["classifications_onet_code"].astype(str).str.strip() != "")
    ].copy()
    if eligible.empty:
        return pd.DataFrame()

    eligible["edu_level"] = eligible["edu_normalized"].map(education_rank)

    flagged_rows: list[dict] = []
    for onet_code, group in eligible.groupby("classifications_onet_code"):
        if len(group) < 3:
            continue

        minimum = group["edu_level"].min()
        for _, row in group.iterrows():
            gap = int(row["edu_level"] - minimum)
            if gap >= 2:
                flagged_rows.append(
                    {
                        "Job Title": row.get("title", ""),
                        "Employer": row.get("application_company", ""),
                        "City": row.get("city", ""),
                        "Required Education": row.get("requirements_min_education", ""),
                        "Minimum in Same Field": inverse_rank.get(int(minimum), "Unknown"),
                        "Gap (levels)": gap,
                        "Education Gap (levels)": gap,
                        "O*NET Code": onet_code,
                        "Salary": format_salary(
                            row.get("salary_min", row.get("parameters_salary_min", "")),
                            row.get("salary_max", row.get("parameters_salary_max", "")),
                        ),
                    }
                )

    if not flagged_rows:
        return pd.DataFrame()

    return pd.DataFrame(flagged_rows).sort_values("Gap (levels)", ascending=False).head(150)


def build_salary_by_city(jobs_clean: pd.DataFrame) -> pd.DataFrame:
    if jobs_clean.empty or "city" not in jobs_clean.columns:
        return pd.DataFrame()

    with_salary = jobs_clean[(jobs_clean.get("salary_min", 0) > 0) & (jobs_clean["city"].astype(str).str.strip() != "")].copy()
    if with_salary.empty:
        return pd.DataFrame()

    city_stats = with_salary.groupby("city", as_index=False).agg(
        avg_min=("salary_min", "mean"),
        median_min=("salary_min", "median"),
        p25=("salary_min", lambda values: np.percentile(values, 25)),
        p75=("salary_min", lambda values: np.percentile(values, 75)),
        max_sal=("salary_min", "max"),
        job_count=("system_job_id", "count"),
    )

    city_stats = city_stats[city_stats["job_count"] >= 3]
    return city_stats.sort_values("avg_min", ascending=False).reset_index(drop=True)


def score_description(
    description: str,
    salary_min,
    salary_max,
    education_requirement: str,
    experience_requirement: str,
) -> tuple[int, dict[str, str]]:
    score = 0
    breakdown: dict[str, str] = {}

    word_count = len(str(description).split())
    if 150 <= word_count <= 600:
        score += 25
        breakdown["Description Length"] = f"Good ({word_count} words). Clear and scannable."
    elif 75 <= word_count < 150 or 600 < word_count <= 900:
        score += 12
        breakdown["Description Length"] = f"Acceptable ({word_count} words). Could be more informative."
    else:
        score += 4
        breakdown["Description Length"] = f"Needs work ({word_count} words). Too short or too long."

    try:
        salary_text = str(salary_min).replace(",", "").strip()
        salary = float(salary_text) if salary_text not in {"", "nan", "0", "0.0"} else 0
        if salary > 0:
            score += 30
            breakdown["Salary Transparency"] = "Salary listed. Improves qualified application volume."
        else:
            breakdown["Salary Transparency"] = "No salary listed. Can reduce application quality."
    except (TypeError, ValueError):
        breakdown["Salary Transparency"] = "No salary listed."

    if str(education_requirement).strip().lower() not in {"", "nan", "none"}:
        score += 15
        breakdown["Education Requirement"] = f"Clearly stated: {education_requirement}"
    else:
        breakdown["Education Requirement"] = "Not specified. Adds applicant uncertainty."

    if str(experience_requirement).strip().lower() not in {"", "nan", "none"}:
        score += 15
        breakdown["Experience Requirement"] = f"Clearly stated: {experience_requirement}"
    else:
        breakdown["Experience Requirement"] = "Not specified. Adds applicant uncertainty."

    quantified = len(re.findall(r"\d+\s*[\+\-]?\s*(?:year|month|week|yr)", str(description).lower()))
    if quantified >= 2:
        score += 15
        breakdown["Specificity"] = f"Contains {quantified} quantified requirements."
    elif quantified == 1:
        score += 8
        breakdown["Specificity"] = "Contains one quantified requirement."
    else:
        breakdown["Specificity"] = "No quantified requirements found."

    return min(score, 100), breakdown