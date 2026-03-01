import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_matching_index(
    skill_profiles: pd.DataFrame,
) -> tuple[TfidfVectorizer, object, list[str]]:
    texts = skill_profiles["skill_text"].fillna("").astype(str).tolist()
    job_ids = skill_profiles["system_job_id"].astype(str).tolist()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, job_ids


def find_matching_jobs(
    user_input: str,
    jobs_clean: pd.DataFrame,
    skill_profiles: pd.DataFrame,
    top_n: int = 8,
    matching_index: tuple[TfidfVectorizer, object, list[str]] | None = None,
) -> pd.DataFrame:
    if skill_profiles.empty:
        return jobs_clean.head(0).copy()

    if matching_index is None:
        vectorizer, matrix, job_ids = build_matching_index(skill_profiles)
    else:
        vectorizer, matrix, job_ids = matching_index

    user_vector = vectorizer.transform([str(user_input)])
    similarities = cosine_similarity(user_vector, matrix)[0]

    top_indices = similarities.argsort()[-top_n:][::-1]
    top_ids = [job_ids[index] for index in top_indices]
    top_scores = [similarities[index] for index in top_indices]

    results = jobs_clean[jobs_clean["system_job_id"].isin(top_ids)].copy()
    score_map = dict(zip(top_ids, top_scores))
    results["match_score"] = results["system_job_id"].map(score_map)

    return results.sort_values("match_score", ascending=False).head(top_n)


def compute_skill_gap(
    user_text: str,
    job_id: str,
    skill_mentions: pd.DataFrame,
    limit: int = 12,
) -> tuple[list[str], list[str]]:
    if skill_mentions.empty:
        return [], []

    required_columns = {"Research ID", "Taxonomy Skill"}
    if not required_columns.issubset(skill_mentions.columns):
        return [], []

    score_column = "NLP Score" if "NLP Score" in skill_mentions.columns else "Correlation Coefficient"
    if score_column not in skill_mentions.columns:
        return [], []

    job_skills = skill_mentions[skill_mentions["Research ID"].astype(str) == str(job_id)]
    if job_skills.empty:
        return [], []

    ranked = (
        job_skills[["Taxonomy Skill", score_column]]
        .copy()
        .sort_values(score_column, ascending=False)
        .drop_duplicates(subset=["Taxonomy Skill"])
        .head(limit)
    )

    user_text_lower = str(user_text).lower()
    matched_skills: list[str] = []
    missing_skills: list[str] = []

    for skill in ranked["Taxonomy Skill"].astype(str).tolist():
        token_candidates = [token for token in skill.lower().split() if len(token) > 3]
        has_match = token_candidates and any(token in user_text_lower for token in token_candidates)
        if has_match:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    return matched_skills, missing_skills
