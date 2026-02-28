import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_matching_jobs(
    user_input: str,
    jobs_clean: pd.DataFrame,
    skill_profiles: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    if skill_profiles.empty:
        return jobs_clean.head(0).copy()

    all_texts = list(skill_profiles["skill_text"].astype(str)) + [str(user_input)]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]

    top_indices = similarities.argsort()[-top_n:][::-1]
    top_ids = [skill_profiles.iloc[index]["system_job_id"] for index in top_indices]
    top_scores = [similarities[index] for index in top_indices]

    results = jobs_clean[jobs_clean["system_job_id"].isin(top_ids)].copy()
    score_map = dict(zip(top_ids, top_scores))
    results["match_score"] = results["system_job_id"].map(score_map)

    return results.sort_values("match_score", ascending=False).head(top_n)
