from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_COLUMNS = [
    "title",
    "description",
    "requirements_min_education",
    "requirements_experience",
    "classifications_onet_code",
]


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def build_job_text_corpus(jobs_clean: pd.DataFrame) -> pd.Series:
    available_cols = [column for column in TEXT_COLUMNS if column in jobs_clean.columns]
    if not available_cols:
        return pd.Series([""] * len(jobs_clean), index=jobs_clean.index)

    corpus = jobs_clean[available_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return corpus.map(_normalize_text)


def build_skill_catalog(
    processed: pd.DataFrame,
    min_frequency: int = 3,
    max_skills: int = 3000,
) -> list[str]:
    if "Taxonomy Skill" not in processed.columns:
        return []

    normalized_skills = (
        processed["Taxonomy Skill"]
        .fillna("")
        .astype(str)
        .map(_normalize_text)
    )

    normalized_skills = normalized_skills[normalized_skills.str.len() > 2]
    counts = normalized_skills.value_counts()

    common_skills = counts[counts >= min_frequency].head(max_skills)
    return common_skills.index.tolist()


def _iter_batches(total_size: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, total_size, batch_size):
        yield start, min(start + batch_size, total_size)


def extract_skill_mentions_from_text(
    jobs_clean: pd.DataFrame,
    processed: pd.DataFrame,
    top_k: int = 15,
    min_similarity: float = 0.08,
    batch_size: int = 256,
) -> pd.DataFrame:
    if jobs_clean.empty:
        return pd.DataFrame(columns=["Research ID", "Taxonomy Skill", "NLP Score"])

    skill_catalog = build_skill_catalog(processed)
    if not skill_catalog:
        return pd.DataFrame(columns=["Research ID", "Taxonomy Skill", "NLP Score"])

    job_corpus = build_job_text_corpus(jobs_clean)

    combined_texts = pd.concat([job_corpus, pd.Series(skill_catalog)], ignore_index=True)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    job_count = len(jobs_clean)
    job_matrix = tfidf_matrix[:job_count]
    skill_matrix = tfidf_matrix[job_count:]

    skill_labels = np.array(skill_catalog)
    job_ids = jobs_clean["system_job_id"].astype(str).tolist()
    records: list[tuple[str, str, float]] = []

    for start, end in _iter_batches(job_count, batch_size):
        batch_similarity = (job_matrix[start:end] @ skill_matrix.T).tocsr()

        for local_row in range(batch_similarity.shape[0]):
            row = batch_similarity.getrow(local_row)
            if row.nnz == 0:
                continue

            scores = row.data
            indices = row.indices

            keep_mask = scores >= min_similarity
            if not np.any(keep_mask):
                continue

            scores = scores[keep_mask]
            indices = indices[keep_mask]

            if len(scores) > top_k:
                top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
                scores = scores[top_idx]
                indices = indices[top_idx]

            order = np.argsort(-scores)
            for score, skill_idx in zip(scores[order], indices[order]):
                records.append(
                    (
                        job_ids[start + local_row],
                        str(skill_labels[skill_idx]),
                        float(score),
                    )
                )

    mentions = pd.DataFrame(records, columns=["Research ID", "Taxonomy Skill", "NLP Score"])
    return mentions


def build_skill_profiles_from_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    if mentions.empty:
        return pd.DataFrame(columns=["system_job_id", "skill_text"])

    ordered = mentions.sort_values(["Research ID", "NLP Score"], ascending=[True, False])
    deduped = ordered.drop_duplicates(subset=["Research ID", "Taxonomy Skill"])

    skill_profiles = (
        deduped.groupby("Research ID", as_index=False)["Taxonomy Skill"]
        .agg(lambda values: " ".join(values.astype(str)))
        .rename(
            columns={
                "Research ID": "system_job_id",
                "Taxonomy Skill": "skill_text",
            }
        )
    )

    return skill_profiles


EDUCATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(ph\.?d|doctorate|doctoral)\b", re.IGNORECASE), "Doctorate"),
    (re.compile(r"\b(master'?s|mba|m\.s\.|m\s?s\b|m\.a\.|m\s?a\b)\b", re.IGNORECASE), "Master's Degree"),
    (re.compile(r"\b(bachelor'?s|b\.s\.|b\s?s\b|b\.a\.|b\s?a\b|undergraduate degree)\b", re.IGNORECASE), "Bachelor's Degree"),
    (re.compile(r"\bassociate'?s\b", re.IGNORECASE), "Associate Degree"),
    (re.compile(r"\b(high school|ged|secondary school)\b", re.IGNORECASE), "High School Diploma/GED"),
    (re.compile(r"\b(certification|certificate)\b", re.IGNORECASE), "Certification"),
]

EXPERIENCE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(\d{1,2})\s*(?:\+|plus)?\s*(?:-|to)?\s*(\d{1,2})?\s*(years|year|yrs|yr|months|month|mos|mo)\b",
        re.IGNORECASE,
    ),
]

ENTRY_LEVEL_PATTERN = re.compile(
    r"\b(entry level|no experience|0\s*years?|fresh graduate|new grad)\b",
    re.IGNORECASE,
)


def _normalize_existing_requirement(value: str) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _infer_education_from_text(text: str) -> str:
    for pattern, label in EDUCATION_PATTERNS:
        if pattern.search(text):
            return label
    return ""


def _infer_experience_from_text(text: str) -> str:
    if ENTRY_LEVEL_PATTERN.search(text):
        return "Entry level / no experience"

    for pattern in EXPERIENCE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        minimum = match.group(1)
        maximum = match.group(2)
        unit = match.group(3).lower()
        normalized_unit = "years" if "year" in unit or "yr" in unit else "months"

        if maximum:
            return f"{minimum}-{maximum} {normalized_unit}"
        return f"{minimum} {normalized_unit}"

    return ""


def infer_education_and_experience(jobs_clean: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "system_job_id",
        "title",
        "description",
        "requirements_min_education",
        "requirements_experience",
    ]
    working = jobs_clean.copy()
    for column in required_columns:
        if column not in working.columns:
            working[column] = ""

    rows: list[dict[str, str]] = []
    for _, row in working.iterrows():
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        raw_education = _normalize_existing_requirement(row.get("requirements_min_education", ""))
        raw_experience = _normalize_existing_requirement(row.get("requirements_experience", ""))

        source_text = f"{title} {description}"

        inferred_education = raw_education or _infer_education_from_text(source_text)
        inferred_experience = raw_experience or _infer_experience_from_text(source_text)

        education_source = "dataset" if raw_education else ("nlp_inferred" if inferred_education else "not_specified")
        experience_source = "dataset" if raw_experience else ("nlp_inferred" if inferred_experience else "not_specified")

        rows.append(
            {
                "system_job_id": str(row.get("system_job_id", "")),
                "education_display": inferred_education,
                "education_source": education_source,
                "experience_display": inferred_experience,
                "experience_source": experience_source,
            }
        )

    return pd.DataFrame(rows)
