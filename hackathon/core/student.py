import pandas as pd


def top_skills(processed: pd.DataFrame, limit: int = 20) -> pd.Series:
    if "Taxonomy Skill" not in processed.columns:
        return pd.Series(dtype="int64")
    return processed["Taxonomy Skill"].value_counts().head(limit)


def top_field_skills(processed: pd.DataFrame, job_ids: list, limit: int = 10) -> pd.Series:
    if "Research ID" not in processed.columns or "Taxonomy Skill" not in processed.columns:
        return pd.Series(dtype="int64")

    field_rows = processed[processed["Research ID"].isin(job_ids)]
    return field_rows["Taxonomy Skill"].value_counts().head(limit)


FIELD_KEYWORDS = {
    "Healthcare & Medicine": "patient care medical nursing health clinical",
    "Technology & IT": "software programming data systems network cloud",
    "Business & Management": "management leadership strategy finance accounting",
    "Engineering": "engineering design systems technical analysis",
    "Education": "teaching curriculum training instruction learning",
    "Logistics & Operations": "logistics supply chain operations warehouse transportation",
}
