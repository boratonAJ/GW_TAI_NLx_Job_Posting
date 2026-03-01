import pandas as pd

from hackathon.core.matching import find_matching_jobs


MOC_DICTIONARY = {
    "11B": ("Infantry Soldier", "leadership team management operations security patrol coordination"),
    "68W": ("Combat Medic", "patient care emergency medical first aid healthcare nursing triage clinical"),
    "25U": (
        "Signal Support Specialist",
        "communications network IT systems radio telecommunications",
    ),
    "92A": (
        "Logistical Specialist",
        "inventory supply chain logistics warehouse procurement management",
    ),
    "31B": ("Military Police", "law enforcement security patrol investigation compliance safety"),
    "25B": ("IT Specialist", "information technology network systems cybersecurity database cloud"),
    "42A": ("Human Resources Specialist", "HR recruiting employee relations administration benefits payroll"),
    "88M": ("Motor Transport Operator", "transportation driving logistics delivery fleet routing"),
    "15T": ("Helicopter Repairer", "aircraft maintenance mechanical aviation repair inspection"),
    "35F": ("Intelligence Analyst", "analysis intelligence data research reporting assessment"),
    "12B": ("Combat Engineer", "construction engineering project management infrastructure planning"),
    "74D": ("CBRN Specialist", "safety environmental hazmat compliance chemical emergency"),
    "56M": ("Chaplain Assistant", "counseling mental health social work community support"),
    "91B": ("Vehicle Mechanic", "automotive mechanical repair maintenance diagnostics fleet"),
    "68C": ("Practical Nursing Specialist", "nursing patient care clinical LPN healthcare treatment"),
    "35L": ("Counterintelligence Agent", "investigation security analysis intelligence law enforcement"),
    "25S": ("Satellite Comm Operator", "telecommunications satellite systems engineering IT network"),
    "13F": ("Fire Support Specialist", "coordination communications analysis operations planning"),
    "19D": ("Cavalry Scout", "reconnaissance operations security intelligence coordination surveillance"),
    "12Y": ("Geospatial Engineer", "GIS mapping data analysis geospatial systems spatial engineering"),
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
    matching_index: tuple | None = None,
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
        matching_index=matching_index,
    )

    return direct_matches, skill_matches, title
