import pandas as pd

from hackathon.core.matching import find_matching_jobs


MOC_DICTIONARY = {
    "11B": ("Infantry", "leadership team management operations security"),
    "68W": ("Combat Medic", "patient care emergency medical first aid healthcare"),
    "25U": (
        "Signal Support Specialist",
        "communications network IT systems technical",
    ),
    "92A": (
        "Automated Logistical Specialist",
        "inventory supply chain logistics warehouse",
    ),
    "31B": ("Military Police", "law enforcement security patrol investigation"),
    "25B": ("IT Specialist", "information technology network systems cybersecurity"),
    "42A": ("Human Resources Specialist", "HR recruiting employee relations administration"),
    "88M": ("Motor Transport Operator", "transportation driving logistics delivery"),
    "15T": ("Helicopter Repairer", "aircraft maintenance mechanical aviation repair"),
    "35F": ("Intelligence Analyst", "analysis data intelligence research reporting"),
    "12B": ("Combat Engineer", "construction engineering project management"),
    "74D": ("CBRN Specialist", "safety environmental hazmat compliance"),
}


def find_direct_moc_matches(moc_code: str, jobs_clean: pd.DataFrame) -> pd.DataFrame:
    code = moc_code.upper().strip()
    if not code:
        return jobs_clean.head(0).copy()

    matches = jobs_clean[jobs_clean["moc_codes"].astype(str).str.upper().str.contains(code, na=False)]
    return matches.copy()


def veteran_full_match(
    moc_code: str,
    jobs_clean: pd.DataFrame,
    skill_profiles: pd.DataFrame,
    top_n: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    code = moc_code.upper().strip()

    direct_matches = find_direct_moc_matches(code, jobs_clean)

    if code in MOC_DICTIONARY:
        title, skill_query = MOC_DICTIONARY[code]
    else:
        title, skill_query = code, "operations leadership management"

    skill_matches = find_matching_jobs(
        user_input=skill_query,
        jobs_clean=jobs_clean,
        skill_profiles=skill_profiles,
        top_n=top_n,
    )

    return direct_matches, skill_matches, title
