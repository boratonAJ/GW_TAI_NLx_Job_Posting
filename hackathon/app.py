import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Colorado Workforce Intelligence",
    page_icon="🏔️",
    layout="wide",
)

from hackathon.core.data import load_data, load_raw_skill_records
from hackathon.core.analytics_logger import (
    analytics_artifact_paths,
    initialize_analytics_logger,
    load_analytics_events,
    log_analytics_event,
)
from hackathon.core.intelligence import (
    analyze_ghost_job_language,
    build_salary_by_city,
    detect_credential_inflation,
    detect_emerging_skills,
    format_salary,
    score_description,
)
from hackathon.core.matching import build_matching_index, compute_skill_gap, find_matching_jobs
from hackathon.core.student import FIELD_KEYWORDS, top_field_skills, top_skills
from hackathon.core.veterans import MOC_DICTIONARY, veteran_full_match


@st.cache_data
def load_cached_data():
    return load_data()


@st.cache_data
def load_cached_raw_processed():
    return load_raw_skill_records()


jobs_clean, skill_profiles, processed = load_cached_data()
processed_raw = load_cached_raw_processed()

jobs_clean["is_ghost"] = jobs_clean.get("ghostjob", "").astype(str).str.lower().isin(["true", "1", "yes", "t"])
jobs_clean["is_federal"] = jobs_clean.get("fedcontractor", "").astype(str).str.lower().isin(["true", "1", "yes", "t"])
jobs_clean["has_apprenticeship"] = jobs_clean.get("rapids_codes", "").astype(str).str.strip().str.len() > 3
jobs_clean["has_moc"] = jobs_clean.get("moc_codes", "").astype(str).str.strip().str.len() > 3
jobs_clean["has_cip"] = jobs_clean.get("cip_codes", "").astype(str).str.strip().str.len() > 3
jobs_clean["salary_min"] = pd.to_numeric(jobs_clean.get("parameters_salary_min", 0), errors="coerce").fillna(0)
jobs_clean["salary_max"] = pd.to_numeric(jobs_clean.get("parameters_salary_max", 0), errors="coerce").fillna(0)
jobs_clean["naics_sector"] = jobs_clean.get("classifications_naics_code", "").astype(str).str[:2]


@st.cache_resource
def load_matching_index(skill_profiles_frame: pd.DataFrame):
    return build_matching_index(skill_profiles_frame)


matching_index = load_matching_index(skill_profiles)


def _format_salary_range(salary_min, salary_max) -> str:
    if not str(salary_min).strip() or not str(salary_max).strip():
        return "Not listed"
    try:
        return f"${float(salary_min):,.0f} - ${float(salary_max):,.0f}"
    except (ValueError, TypeError):
        return "Not listed"


def _demand_tier_by_rank(rank: int) -> str:
    if rank <= 6:
        return "High"
    if rank <= 14:
        return "Medium"
    return "Low"


def _style_demand_tier_row(row: pd.Series) -> list[str]:
    tier = str(row.get("Demand Tier", ""))
    tier_color_map = {
        "High": "background-color: rgba(34, 197, 94, 0.20); color: #14532d; font-weight: 700;",
        "Medium": "background-color: rgba(245, 158, 11, 0.20); color: #78350f; font-weight: 700;",
        "Low": "background-color: rgba(148, 163, 184, 0.20); color: #334155; font-weight: 700;",
    }
    style_for_row = [""] * len(row)
    demand_tier_index = row.index.get_loc("Demand Tier")
    style_for_row[demand_tier_index] = tier_color_map.get(tier, "")
    return style_for_row


def _init_analytics_state() -> None:
    initialize_analytics_logger(PROJECT_ROOT)

    if "analytics_initialized" not in st.session_state:
        st.session_state.analytics_initialized = True
        st.session_state.visit_timestamps = []
        st.session_state.job_search_events = []
        st.session_state.field_search_events = []
        st.session_state.veteran_search_events = []
        st.session_state.recommendation_events = []

    if "visit_logged_for_session" not in st.session_state:
        now_iso = datetime.now().isoformat()
        st.session_state.visit_logged_for_session = True
        st.session_state.visit_timestamps.append(now_iso)
        log_analytics_event(
            PROJECT_ROOT,
            {
                "timestamp": now_iso,
                "event_type": "visit",
                "channel": "app",
            },
        )


def _log_recommendations(results: pd.DataFrame, channel: str, top_n: int = 5) -> None:
    if results.empty:
        return

    timestamp = datetime.now().isoformat()
    for _, row in results.head(top_n).iterrows():
        st.session_state.recommendation_events.append(
            {
                "timestamp": timestamp,
                "channel": channel,
                "title": str(row.get("title", "Untitled")),
                "city": str(row.get("city", "Unknown City")),
                "match_score": float(row.get("match_score", 0) or 0),
            }
        )
        log_analytics_event(
            PROJECT_ROOT,
            {
                "timestamp": timestamp,
                "event_type": "recommendation",
                "channel": channel,
                "title": str(row.get("title", "Untitled")),
                "city": str(row.get("city", "Unknown City")),
                "match_score": float(row.get("match_score", 0) or 0),
            },
        )


_init_analytics_state()


FIELD_DESCRIPTIONS = {
    "Healthcare & Medicine": "Clinical care, patient support, public health, and treatment-focused roles.",
    "Technology & IT": "Software, data, infrastructure, cloud, and cybersecurity-oriented careers.",
    "Business & Management": "Operations, finance, strategy, people leadership, and administration tracks.",
    "Engineering": "Design, systems, technical analysis, and infrastructure-driven opportunities.",
    "Education": "Teaching, curriculum development, instruction, and workforce training roles.",
    "Logistics & Operations": "Supply chain, transportation, warehousing, and process coordination work.",
}


st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 12%, rgba(56, 189, 248, 0.16), transparent 34%),
            radial-gradient(circle at 82% 18%, rgba(14, 165, 233, 0.12), transparent 30%),
            linear-gradient(135deg, #0b1220 0%, #111827 42%, #0f172a 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            repeating-linear-gradient(
                90deg,
                rgba(148, 163, 184, 0.06) 0px,
                rgba(148, 163, 184, 0.06) 1px,
                transparent 1px,
                transparent 68px
            ),
            repeating-linear-gradient(
                0deg,
                rgba(148, 163, 184, 0.05) 0px,
                rgba(148, 163, 184, 0.05) 1px,
                transparent 1px,
                transparent 68px
            );
        opacity: 0.18;
        z-index: 0;
    }
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1160px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        border-radius: 8px;
        padding: 0.45rem 0.8rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(31, 41, 55, 0.08);
    }
    .hero-wrap {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(56, 189, 248, 0.28);
        border-radius: 10px;
        padding: 0.95rem 1rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(4px);
    }
    .result-card {
        border: 1px solid rgba(71, 85, 105, 0.6);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        margin-bottom: 0.55rem;
        background: rgba(15, 23, 42, 0.48);
    }
    .skill-card {
        border: 1px solid rgba(71, 85, 105, 0.6);
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        margin-bottom: 0.45rem;
        background: rgba(15, 23, 42, 0.52);
    }
    .section-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: rgba(125, 211, 252, 0.95);
        text-transform: uppercase;
        margin-bottom: 0.15rem;
    }
    .subtle {
        color: rgba(226, 232, 240, 0.9);
        font-size: 0.92rem;
        margin-top: 0;
        margin-bottom: 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Colorado Workforce Intelligence")
st.caption("Powered by NLx data with NLP-extracted skill intelligence")
st.markdown(
    f"""
<div class="hero-wrap">
  <div class="section-kicker">Workforce Discovery Dashboard</div>
  <p class="subtle">Explore {len(jobs_clean):,} Colorado postings with skill, education, and experience signals extracted for transparent matching.</p>
</div>
""",
    unsafe_allow_html=True,
)

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
city_count = jobs_clean["city"].astype(str).str.strip()
city_count = city_count[city_count != ""].nunique()
employer_count = jobs_clean["application_company"].astype(str).str.strip()
employer_count = employer_count[employer_count != ""].nunique()
overview_col1.metric("Jobs Indexed", f"{len(jobs_clean):,}")
overview_col2.metric("Cities Covered", f"{city_count:,}")
overview_col3.metric("Skill Profiles", f"{len(skill_profiles):,}")
overview_col4.metric("Employers", f"{employer_count:,}")
st.markdown("---")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Job Seeker",
        "Student / Career Changer",
        "Veteran",
        "Market Intelligence",
        "Recruiter Tools",
        "Usage Insights",
    ]
)

with tab1:
    st.header("Find Jobs That Match Your Skills")
    st.caption("Describe what you can do in plain language, then apply precision filters for cleaner matches.")

    input_col, filter_col = st.columns([2, 1])
    with input_col:
        user_skills = st.text_area(
            "Your skills and experience",
            placeholder=(
                "Example: I have customer service experience, Excel skills, "
                "and enjoy solving problems with people."
            ),
            height=130,
        )
    with filter_col:
        city_options = ["All Cities"] + sorted([city for city in jobs_clean["city"].unique() if str(city).strip()])
        city_filter = st.selectbox("Filter by city", city_options)
        num_results = st.slider("Number of results", min_value=5, max_value=20, value=10, step=1)
        hide_ghosts = st.checkbox("Hide ghost jobs", value=True)
        federal_only = st.checkbox("Federal contractor jobs only", value=False)
        apprenticeship_only = st.checkbox("Apprenticeship pathways only", value=False)
        run_search = st.button("Find Matching Jobs", type="primary", use_container_width=True)

    if run_search:
        if not user_skills.strip():
            st.warning("Please describe your skills first.")
        else:
            with st.spinner("Matching your skills to Colorado jobs..."):
                results = find_matching_jobs(
                    user_skills,
                    jobs_clean,
                    skill_profiles,
                    top_n=min(num_results * 4, 80),
                    matching_index=matching_index,
                )

                if city_filter != "All Cities":
                    results = results[results["city"] == city_filter]
                if hide_ghosts:
                    results = results[~results["is_ghost"]]
                if federal_only:
                    results = results[results["is_federal"]]
                if apprenticeship_only:
                    results = results[results["has_apprenticeship"]]

                results = results.head(num_results)

            st.session_state.job_search_events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "city_filter": city_filter,
                    "results_count": int(len(results)),
                }
            )
            log_analytics_event(
                PROJECT_ROOT,
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "search",
                    "channel": "job_seeker",
                    "city_filter": city_filter,
                    "results_count": int(len(results)),
                },
            )
            _log_recommendations(results, channel="job_seeker", top_n=5)

            st.success(f"Found {len(results)} matching jobs")

            if results.empty:
                st.info("No matches found for the selected filters. Try broader skills or remove city filtering.")
            elif hide_ghosts:
                ghost_count = int(jobs_clean["is_ghost"].sum())
                st.caption(f"Ghost-job filter is active. {ghost_count:,} flagged postings were excluded.")

            for _, row in results.iterrows():
                title = row.get("title", "Untitled Role")
                company = row.get("application_company", "Unknown Company")
                city = row.get("city", "Unknown City")
                score = row.get("match_score", 0)
                score_label = "Strong" if float(score) >= 0.50 else "Moderate" if float(score) >= 0.30 else "Partial"
                badge_parts = [f"{float(score):.0%} {score_label}"]
                if row.get("is_ghost", False):
                    badge_parts.append("Ghost")
                if row.get("is_federal", False):
                    badge_parts.append("Federal")
                if row.get("has_apprenticeship", False):
                    badge_parts.append("Apprenticeship")

                with st.expander(f"{title} · {company} · {city} · {' | '.join(badge_parts)}"):
                    score = row.get("match_score", 0)
                    top_l, top_r, top_rr = st.columns([1, 1, 1])
                    top_l.metric("Match Score", f"{float(score):.0%}")
                    top_r.metric("City", str(city) if str(city).strip() else "Unknown")
                    top_rr.metric(
                        "Salary",
                        _format_salary_range(
                            row.get("parameters_salary_min", ""),
                            row.get("parameters_salary_max", ""),
                        ),
                    )
                    education_value = row.get("education_display", "") or "Not specified"
                    education_source = row.get("education_source", "not_specified")
                    education_status = "Specified" if education_source != "not_specified" else "Not specified"
                    education_source_label = (
                        "Dataset"
                        if education_source == "dataset"
                        else "NLP inferred"
                        if education_source == "nlp_inferred"
                        else "None"
                    )
                    st.write(f"Education: {education_value} ({education_status}, source: {education_source_label})")

                    experience_value = row.get("experience_display", "") or "Not specified"
                    experience_source = row.get("experience_source", "not_specified")
                    experience_status = "Specified" if experience_source != "not_specified" else "Not specified"
                    experience_source_label = (
                        "Dataset"
                        if experience_source == "dataset"
                        else "NLP inferred"
                        if experience_source == "nlp_inferred"
                        else "None"
                    )
                    st.write(f"Experience: {experience_value} ({experience_status}, source: {experience_source_label})")

                    matched_skills, missing_skills = compute_skill_gap(
                        user_text=user_skills,
                        job_id=row.get("system_job_id", ""),
                        skill_mentions=processed,
                        limit=10,
                    )

                    gap_left, gap_right = st.columns(2)
                    with gap_left:
                        st.caption("Matched Skills")
                        if matched_skills:
                            for skill_name in matched_skills[:5]:
                                st.write(f"• {skill_name}")
                        else:
                            st.write("• No strong direct skill matches found")

                    with gap_right:
                        st.caption("Potential Skill Gaps")
                        if missing_skills:
                            for skill_name in missing_skills[:5]:
                                st.write(f"• {skill_name}")
                        else:
                            st.write("• No major gaps detected in top skills")

                    job_link = str(row.get("link", "")).strip()
                    if job_link:
                        st.link_button("View Job Posting", job_link, use_container_width=True)

    st.markdown("---")
    st.subheader("Discover Adjacent Roles")
    st.caption("Surface skill-adjacent jobs across industries that may not match your original title expectations.")
    adjacent_text = st.text_area(
        "Background for adjacent role discovery",
        height=90,
        key="adjacent_text",
        placeholder="Example: project coordination, stakeholder communication, reporting, operations support",
    )
    if st.button("Find Adjacent Careers", key="adjacent_btn"):
        if not adjacent_text.strip():
            st.warning("Please provide a short background description first.")
        else:
            adjacent_results = find_matching_jobs(
                adjacent_text,
                jobs_clean,
                skill_profiles,
                top_n=30,
                matching_index=matching_index,
            )
            if adjacent_results.empty:
                st.info("No adjacent career matches found right now.")
            else:
                seen_sectors = set()
                selected_rows = []
                for _, adjacent_row in adjacent_results.iterrows():
                    sector = str(adjacent_row.get("naics_sector", ""))
                    if sector not in seen_sectors and len(selected_rows) < 8:
                        seen_sectors.add(sector)
                        selected_rows.append(adjacent_row)
                adjacent_display = pd.DataFrame(
                    {
                        "Job Title": [r.get("title", "") for r in selected_rows],
                        "Employer": [r.get("application_company", "") for r in selected_rows],
                        "City": [r.get("city", "") for r in selected_rows],
                        "Match": [f"{float(r.get('match_score', 0)):.0%}" for r in selected_rows],
                        "Salary": [
                            format_salary(
                                r.get("salary_min", r.get("parameters_salary_min", "")),
                                r.get("salary_max", r.get("parameters_salary_max", "")),
                                r.get("parameters_salary_unit", ""),
                            )
                            for r in selected_rows
                        ],
                    }
                )
                st.dataframe(adjacent_display, hide_index=True, use_container_width=True)

with tab2:
    st.header("What Skills Are Trending in Colorado?")
    st.caption("See the most demanded skills and explore likely roles by field.")

    skill_counts = top_skills(processed, limit=20)
    skill_df = skill_counts.reset_index()
    skill_df.columns = ["Skill", "Mentions"]
    skill_df["Rank"] = range(1, len(skill_df) + 1)
    skill_df["Demand Tier"] = skill_df["Rank"].map(_demand_tier_by_rank)
    total_mentions_top20 = int(skill_df["Mentions"].sum()) if not skill_df.empty else 0

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    if not skill_df.empty:
        top_skill = skill_df.iloc[0]
        metric_col1.metric("Most In-Demand Skill", str(top_skill["Skill"]))
        metric_col2.metric("Top Skill Mentions", f"{int(top_skill['Mentions']):,}")
        metric_col3.metric("Top-20 Total Mentions", f"{total_mentions_top20:,}")
    else:
        metric_col1.metric("Most In-Demand Skill", "N/A")
        metric_col2.metric("Top Skill Mentions", "0")
        metric_col3.metric("Top-20 Total Mentions", "0")

    left_panel, right_panel = st.columns([1.55, 1])
    with left_panel:
        st.subheader("Top 20 In-Demand Skills")
        if not skill_counts.empty:
            st.bar_chart(skill_counts.sort_values(ascending=True))
            table_df = skill_df[["Rank", "Skill", "Mentions", "Demand Tier"]].copy()
            styled_table = table_df.style.apply(_style_demand_tier_row, axis=1)
            st.dataframe(styled_table, hide_index=True, use_container_width=True)
        else:
            st.info("No skill demand data available yet.")

        st.markdown("---")
        st.subheader("Skills by Taxonomy Source")
        taxonomy_required = {"Taxonomy Source", "Taxonomy Skill"}
        if taxonomy_required.issubset(processed_raw.columns):
            source_left, source_right = st.columns(2)
            with source_left:
                esco_series = processed_raw[
                    processed_raw["Taxonomy Source"].astype(str).str.upper().str.contains("ESCO", na=False)
                ]["Taxonomy Skill"].value_counts().head(15)
                if len(esco_series) > 0:
                    esco_df = esco_series.reset_index()
                    esco_df.columns = ["Skill", "Count"]
                    st.markdown("**ESCO Taxonomy**")
                    st.dataframe(esco_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No ESCO-tagged taxonomy rows found.")
            with source_right:
                onet_series = processed_raw[
                    processed_raw["Taxonomy Source"].astype(str).str.upper().str.contains(r"ONET|O\\.NET", na=False, regex=True)
                ]["Taxonomy Skill"].value_counts().head(15)
                if len(onet_series) > 0:
                    onet_df = onet_series.reset_index()
                    onet_df.columns = ["Skill", "Count"]
                    st.markdown("**O*NET Taxonomy**")
                    st.dataframe(onet_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No O*NET-tagged taxonomy rows found.")
        else:
            st.info("Taxonomy source columns are unavailable in the current raw processed dataset.")

    with right_panel:
        st.subheader("Top Skill Snapshot")
        for rank, row in enumerate(skill_df.head(5).itertuples(index=False), start=1):
            mentions = int(row.Mentions)
            share = (mentions / total_mentions_top20 * 100) if total_mentions_top20 else 0
            st.markdown(
                f"""
<div class="skill-card">
  <div class="section-kicker">Rank #{rank}</div>
  <div><strong>{row.Skill}</strong></div>
  <p class="subtle">{mentions:,} mentions • {share:.1f}% of top-20 demand</p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Career Field Explorer")
        field = st.selectbox(
            "I am interested in",
            list(FIELD_KEYWORDS.keys()),
            help="Select a pathway to view role matches and top skills to build.",
        )
        st.caption(FIELD_DESCRIPTIONS.get(field, "Explore job and skill demand for this field."))
        run_field = st.button("Show Field Insights", type="primary", use_container_width=True)

    if run_field:
        with st.spinner("Analyzing this career field..."):
            query = FIELD_KEYWORDS[field]
            results = find_matching_jobs(
                query,
                jobs_clean,
                skill_profiles,
                top_n=8,
                matching_index=matching_index,
            )
            field_job_ids = results["system_job_id"].tolist()
            field_skill_counts = top_field_skills(processed, field_job_ids, limit=10)

        st.session_state.field_search_events.append(
            {
                "timestamp": datetime.now().isoformat(),
                "field": field,
                "results_count": int(len(results)),
            }
        )
        log_analytics_event(
            PROJECT_ROOT,
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": "search",
                "channel": "student_field",
                "field": field,
                "results_count": int(len(results)),
            },
        )
        _log_recommendations(results, channel="student_field", top_n=5)

        st.success(f"Top roles and skills for {field}")

        summary_col1, summary_col2 = st.columns(2)
        summary_col1.metric("Matched Roles", f"{len(results):,}")
        summary_col2.metric("Top Skills Shown", f"{len(field_skill_counts):,}")

        view_mode = st.radio(
            "View Mode",
            options=["Table View", "Card View"],
            horizontal=True,
            label_visibility="collapsed",
        )

        left, right = st.columns(2)
        with left:
            st.subheader("Skills to Learn")
            if field_skill_counts.empty:
                st.info("No field-specific skills found for this selection.")
            else:
                skill_table = field_skill_counts.reset_index()
                skill_table.columns = ["Skill", "Demand Count"]
                skill_table.insert(0, "Rank", range(1, len(skill_table) + 1))
                if view_mode == "Table View":
                    st.dataframe(skill_table, hide_index=True, use_container_width=True)
                else:
                    for _, skill_row in skill_table.iterrows():
                        st.markdown(
                            f"""
<div class="skill-card">
  <div class="section-kicker">Rank #{int(skill_row['Rank'])}</div>
  <div><strong>{skill_row['Skill']}</strong></div>
  <p class="subtle">Demand count: {int(skill_row['Demand Count']):,}</p>
</div>
""",
                            unsafe_allow_html=True,
                        )

        with right:
            st.subheader("Sample Jobs")
            if results.empty:
                st.info("No job matches available for this field right now.")
            else:
                jobs_table = results.head(8).copy()
                jobs_table = jobs_table[
                    ["title", "city", "application_company", "match_score"]
                ].rename(
                    columns={
                        "title": "Job Title",
                        "city": "City",
                        "application_company": "Company",
                        "match_score": "Match Score",
                    }
                )
                jobs_table["Match Score"] = jobs_table["Match Score"].fillna(0).map(lambda value: f"{float(value):.0%}")
                if view_mode == "Table View":
                    st.dataframe(jobs_table, hide_index=True, use_container_width=True)
                else:
                    for job_row in jobs_table.itertuples(index=False):
                        st.markdown(
                            f"""
<div class="result-card">
  <div><strong>{job_row[0]}</strong></div>
  <p class="subtle">{job_row[2]} • {job_row[1]} • Match: {job_row[3]}</p>
</div>
""",
                            unsafe_allow_html=True,
                        )

    st.markdown("---")
    st.subheader("Emerging Skills Detector")
    st.caption("Find low-confidence raw skill phrases that are appearing across employers before taxonomy catches up.")
    emerging_col1, emerging_col2 = st.columns(2)
    with emerging_col1:
        student_confidence_threshold = st.slider(
            "Taxonomy confidence threshold",
            0.40,
            0.80,
            0.65,
            0.05,
            key="student_emerging_confidence",
        )
    with emerging_col2:
        student_min_employers = st.slider(
            "Minimum employers requesting skill",
            1,
            10,
            2,
            1,
            key="student_emerging_employers",
        )

    if st.button("Detect Emerging Skills", type="primary", key="student_emerging_btn"):
        student_emerging = detect_emerging_skills(
            processed_raw,
            confidence_threshold=student_confidence_threshold,
            min_employers=student_min_employers,
            top_n=30,
        )
        if student_emerging.empty:
            st.info("No emerging skills detected at current threshold settings.")
        else:
            display_columns = {
                "Raw Skill": "What Employer Actually Wrote",
                "employer_count": "Employers Requesting This",
                "avg_confidence": "Taxonomy Confidence",
                "closest_taxonomy": "Closest Official Category",
                "taxonomy_source": "Taxonomy Source",
            }
            st.dataframe(
                student_emerging.rename(columns=display_columns).round({"Taxonomy Confidence": 3}),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("No-Degree Pathways")
    st.caption("Registered apprenticeship pathways from RAPIDS-linked postings in Colorado.")
    apprenticeship_jobs = jobs_clean[jobs_clean["has_apprenticeship"]].copy()
    apprenticeship_salary = apprenticeship_jobs[apprenticeship_jobs["salary_min"] > 0]["salary_min"]

    apprenticeship_metric_1, apprenticeship_metric_2 = st.columns(2)
    apprenticeship_metric_1.metric("Jobs With Apprenticeship Path", f"{len(apprenticeship_jobs):,}")
    apprenticeship_metric_2.metric(
        "Avg Minimum Salary",
        f"${apprenticeship_salary.mean():,.0f}" if not apprenticeship_salary.empty else "N/A",
    )

    if apprenticeship_jobs.empty:
        st.info("No apprenticeship-linked postings are available in the current dataset snapshot.")
    else:
        apprenticeship_table = apprenticeship_jobs[
            [
                "title",
                "city",
                "application_company",
                "salary_min",
                "salary_max",
                "requirements_min_education",
                "rapids_codes",
            ]
        ].copy()
        apprenticeship_table["salary_min"] = apprenticeship_table["salary_min"].apply(
            lambda value: f"${value:,.0f}" if float(value) > 0 else "Not listed"
        )
        apprenticeship_table["salary_max"] = apprenticeship_table["salary_max"].apply(
            lambda value: f"${value:,.0f}" if float(value) > 0 else "Not listed"
        )
        apprenticeship_table.columns = [
            "Job Title",
            "City",
            "Employer",
            "Min Salary",
            "Max Salary",
            "Education Required",
            "RAPIDS Code",
        ]
        st.dataframe(apprenticeship_table, hide_index=True, use_container_width=True)

with tab3:
    st.header("Veteran Career Translator")
    st.caption("Translate military occupational experience into civilian Colorado opportunities using direct MOC mapping plus NLP skill matching.")

    col1, col2 = st.columns([1, 2])

    with col1:
        moc_input = st.text_input("Enter MOS/MOC code:", placeholder="e.g. 68W, 11B, 25B").upper().strip()
        veteran_text = st.text_area(
            "Describe your military experience (optional but recommended)",
            height=120,
            placeholder="Example: Led a 12-person logistics team, managed inventory, coordinated transport operations.",
            key="veteran_free_text",
        )
        veteran_federal_only = st.checkbox("Federal contractor jobs only", value=False, key="veteran_federal_only")
        st.markdown("**Common MOS/MOC Codes**")
        for code, (role, _) in list(MOC_DICTIONARY.items())[:8]:
            st.caption(f"{code} — {role}")

    with col2:
        if st.button("Find Civilian Career Matches", type="primary", use_container_width=True):
            if not moc_input and not veteran_text.strip():
                st.warning("Please enter a MOS/MOC code or describe your military experience.")
            else:
                with st.spinner("Translating military experience..."):
                    direct_matches = jobs_clean.head(0).copy()
                    moc_title = "Custom Profile"
                    base_query = "operations leadership management team coordination"

                    if moc_input:
                        direct_matches, skill_matches, moc_title = veteran_full_match(
                            moc_input,
                            jobs_clean,
                            skill_profiles,
                            top_n=30,
                            matching_index=matching_index,
                        )
                        if moc_input in MOC_DICTIONARY:
                            base_query = MOC_DICTIONARY[moc_input][1]
                    else:
                        skill_matches = find_matching_jobs(
                            user_input=veteran_text,
                            jobs_clean=jobs_clean,
                            skill_profiles=skill_profiles,
                            top_n=30,
                            matching_index=matching_index,
                        )

                    combined_query = " ".join([base_query, veteran_text]).strip()
                    if combined_query:
                        skill_matches = find_matching_jobs(
                            user_input=combined_query,
                            jobs_clean=jobs_clean,
                            skill_profiles=skill_profiles,
                            top_n=30,
                            matching_index=matching_index,
                        )

                    if veteran_federal_only:
                        direct_matches = direct_matches[direct_matches["is_federal"]]
                        skill_matches = skill_matches[skill_matches["is_federal"]]

                    skill_matches = skill_matches.head(10)

                st.session_state.veteran_search_events.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "moc": moc_input,
                        "direct_count": int(len(direct_matches)),
                        "skill_count": int(len(skill_matches)),
                    }
                )
                log_analytics_event(
                    PROJECT_ROOT,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "search",
                        "channel": "veteran",
                        "moc": moc_input,
                        "direct_count": int(len(direct_matches)),
                        "skill_count": int(len(skill_matches)),
                        "results_count": int(len(skill_matches)),
                    },
                )
                _log_recommendations(skill_matches, channel="veteran", top_n=5)

                results_label = f"{moc_input} — {moc_title}" if moc_input else "Custom military profile"
                st.success(f"Results for {results_label}")

                if not direct_matches.empty:
                    st.subheader("Stage 1 — Direct MOC-Tagged Matches")
                    for _, row in direct_matches.head(6).iterrows():
                        with st.expander(
                            f"{row.get('title', 'Untitled')} · {row.get('application_company', 'Unknown Company')} · {row.get('city', 'Unknown City')}"
                        ):
                            d1, d2 = st.columns(2)
                            d1.write(
                                f"Salary: {format_salary(row.get('salary_min', 0), row.get('salary_max', 0), row.get('parameters_salary_unit', ''))}"
                            )
                            d2.write(
                                "Education: "
                                f"{row.get('requirements_min_education', 'Not specified') or 'Not specified'}"
                            )
                            job_link = str(row.get("link", "")).strip()
                            if job_link:
                                st.link_button("View Job Posting", job_link, use_container_width=True)

                st.subheader("Stage 2 — Skill-Based Matches")
                for _, row in skill_matches.head(6).iterrows():
                    title = row.get("title", "Untitled")
                    city = row.get("city", "Unknown City")
                    score = float(row.get("match_score", 0))
                    with st.expander(f"{title} — {city} ({score:.0%} match)"):
                        st.write(f"Company: {row.get('application_company', 'Unknown Company')}")
                        st.write(
                            "Education Required: "
                            f"{row.get('requirements_min_education', 'Not specified') or 'Not specified'}"
                        )
                        if row.get("is_federal", False):
                            st.caption("Federal contractor role — veteran hiring preference may apply.")

                        matched_skills, missing_skills = compute_skill_gap(
                            user_text=combined_query,
                            job_id=row.get("system_job_id", ""),
                            skill_mentions=processed,
                            limit=10,
                        )

                        left_gap, right_gap = st.columns(2)
                        with left_gap:
                            st.caption("Transferable Skills")
                            if matched_skills:
                                for skill_name in matched_skills[:5]:
                                    st.write(f"• {skill_name}")
                            else:
                                st.write("• None identified in top taxonomy skills")
                        with right_gap:
                            st.caption("Skills to Bridge")
                            if missing_skills:
                                for skill_name in missing_skills[:5]:
                                    st.write(f"• {skill_name}")
                            else:
                                st.write("• No major gaps detected")

                        job_link = str(row.get("link", "")).strip()
                        if job_link:
                            st.link_button("View Job Posting", job_link, use_container_width=True)

with tab4:
    st.header("Market Intelligence")
    st.caption("Labor market diagnostics from NLx postings: credential inflation, salary fairness, ghost language, and education-to-job alignment.")

    mi_tab1, mi_tab2, mi_tab3, mi_tab4 = st.tabs(
        ["Credential Inflation", "Salary Fairness", "Ghost Job Analysis", "Education-Job Alignment"]
    )

    with mi_tab1:
        st.subheader("Credential Inflation Scanner")
        if st.button("Run Credential Inflation Scan", type="primary", key="mi_credential"):
            ci_df = detect_credential_inflation(jobs_clean)
            if ci_df.empty:
                st.info("No significant credential inflation pattern detected from current eligible postings.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Flagged Postings", f"{len(ci_df):,}")
                c2.metric("Avg Gap", f"{ci_df['Education Gap (levels)'].mean():.1f} levels")
                c3.metric("O*NET Groups", f"{ci_df['O*NET Code'].nunique():,}")
                st.dataframe(ci_df, hide_index=True, use_container_width=True)

    with mi_tab2:
        st.subheader("Salary Fairness by City")
        city_salary = build_salary_by_city(jobs_clean)
        if city_salary.empty:
            st.info("Insufficient salary data to build city-level benchmarks.")
        else:
            st.bar_chart(city_salary.set_index("city")["avg_min"].head(25))
            st.dataframe(city_salary, hide_index=True, use_container_width=True)

        st.markdown("---")
        st.subheader("Role-Specific Salary Lookup")
        salary_lookup_query = st.text_input(
            "Role to benchmark",
            placeholder="e.g., data analyst, registered nurse, software engineer",
            key="mi_salary_lookup_query",
        )
        salary_lookup_city = st.selectbox(
            "City",
            ["All Cities"] + sorted([city for city in jobs_clean["city"].astype(str).unique() if city.strip()]),
            key="mi_salary_lookup_city",
        )

        if st.button("Look Up Salary", type="primary", key="mi_salary_lookup_btn"):
            if not salary_lookup_query.strip():
                st.warning("Please enter a role to benchmark.")
            else:
                salary_results = find_matching_jobs(
                    salary_lookup_query,
                    jobs_clean,
                    skill_profiles,
                    top_n=60,
                    matching_index=matching_index,
                )
                if salary_lookup_city != "All Cities":
                    salary_results = salary_results[salary_results["city"] == salary_lookup_city]

                salary_sample = salary_results[salary_results["salary_min"] > 0]
                if len(salary_sample) < 3:
                    st.info("Insufficient salary data for this role and city selection.")
                else:
                    salary_values = salary_sample["salary_min"].tolist()
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Postings Analyzed", f"{len(salary_sample):,}")
                    s2.metric("Market Average", f"${np.mean(salary_values):,.0f}")
                    s3.metric("25th Percentile", f"${np.percentile(salary_values, 25):,.0f}")
                    s4.metric("75th Percentile", f"${np.percentile(salary_values, 75):,.0f}")

                    by_city = salary_sample.groupby("city")["salary_min"].mean().sort_values(ascending=False).head(15)
                    if len(by_city) > 1:
                        st.bar_chart(by_city)

    with mi_tab3:
        st.subheader("Ghost Job Language Analysis")
        ghost_terms, real_terms = analyze_ghost_job_language(jobs_clean)
        if ghost_terms is None or real_terms is None:
            st.info("Insufficient ghost/real description data for robust language comparison.")
        else:
            left, right = st.columns(2)
            with left:
                st.markdown("**Terms More Common in Ghost Postings**")
                st.dataframe(ghost_terms.round(4), hide_index=True, use_container_width=True)
            with right:
                st.markdown("**Terms More Common in Real Postings**")
                st.dataframe(real_terms.round(4), hide_index=True, use_container_width=True)

    with mi_tab4:
        st.subheader("Education Program vs Job Market Alignment")
        cip_jobs = jobs_clean[jobs_clean["has_cip"]]

        a1, a2 = st.columns(2)
        a1.metric("Postings Linked to CIP Programs", f"{len(cip_jobs):,}")
        a2.metric("Postings Without CIP Codes", f"{len(jobs_clean) - len(cip_jobs):,}")

        if cip_jobs.empty:
            st.info("No CIP-linked postings were found in the current dataset snapshot.")
        else:
            all_cip_codes: list[str] = []
            for value in cip_jobs["cip_codes"].dropna():
                split_codes = str(value).replace("[", "").replace("]", "").replace("'", "").split(",")
                all_cip_codes.extend([code.strip() for code in split_codes if code.strip() and code.strip() != "nan"])

            cip_counts = pd.Series(all_cip_codes).value_counts().head(25)
            cip_table = cip_counts.reset_index()
            cip_table.columns = ["CIP Code", "Employer Count"]

            st.bar_chart(cip_counts.sort_values(ascending=True))
            st.dataframe(cip_table, hide_index=True, use_container_width=True)

with tab5:
    st.header("Recruiter Tools")
    st.caption("Improve job posting quality and benchmark salary offers with NLx market signals.")

    rec_tab1, rec_tab2 = st.tabs(["Description Quality", "Salary Benchmark"])

    with rec_tab1:
        description_text = st.text_area(
            "Paste job description",
            height=180,
            placeholder="Paste full description, requirements, and responsibilities.",
        )
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            education_req = st.text_input("Education requirement", placeholder="e.g., Bachelor's Degree")
        with rc2:
            experience_req = st.text_input("Experience requirement", placeholder="e.g., 3-5 years")
        with rc3:
            planned_min = st.text_input("Planned min salary", placeholder="e.g., 65000")
        with rc4:
            planned_max = st.text_input("Planned max salary", placeholder="e.g., 90000")

        if st.button("Score Description", type="primary", key="rec_score"):
            if not description_text.strip():
                st.warning("Please paste a description to score.")
            else:
                score, breakdown = score_description(
                    description_text,
                    planned_min,
                    planned_max,
                    education_req,
                    experience_req,
                )
                s1, s2 = st.columns([1, 2])
                s1.metric("Quality Score", f"{score}/100")
                with s2:
                    for label, detail in breakdown.items():
                        st.write(f"**{label}:** {detail}")

    with rec_tab2:
        benchmark_query = st.text_input("Role or skill profile", placeholder="e.g., data analyst, registered nurse")
        benchmark_city = st.selectbox(
            "Benchmark city",
            ["All Cities"] + sorted([city for city in jobs_clean["city"].astype(str).unique() if city.strip()]),
            key="rec_city",
        )
        offered_salary = st.number_input("Planned minimum salary ($)", min_value=0, max_value=500000, value=0, step=5000)

        if st.button("Run Salary Benchmark", type="primary", key="rec_benchmark"):
            if not benchmark_query.strip():
                st.warning("Please enter a role or skill query.")
            else:
                benchmark_results = find_matching_jobs(
                    benchmark_query,
                    jobs_clean,
                    skill_profiles,
                    top_n=60,
                    matching_index=matching_index,
                )
                if benchmark_city != "All Cities":
                    benchmark_results = benchmark_results[benchmark_results["city"] == benchmark_city]

                with_salary = benchmark_results[benchmark_results["salary_min"] > 0]
                if len(with_salary) < 3:
                    st.info("Insufficient salary data for robust benchmarking in this segment.")
                else:
                    salaries = with_salary["salary_min"].tolist()
                    p25 = np.percentile(salaries, 25)
                    p50 = np.percentile(salaries, 50)
                    p75 = np.percentile(salaries, 75)

                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Comparable Postings", f"{len(with_salary):,}")
                    b2.metric("Median", f"${p50:,.0f}")
                    b3.metric("25th Percentile", f"${p25:,.0f}")
                    b4.metric("75th Percentile", f"${p75:,.0f}")

                    if offered_salary > 0:
                        if offered_salary >= p75:
                            position = "Top quartile"
                        elif offered_salary >= p50:
                            position = "Above median"
                        elif offered_salary >= p25:
                            position = "Below median"
                        else:
                            position = "Bottom quartile"
                        st.info(f"Planned salary positioning: {position} for this benchmark set.")

                    preview = with_salary[["title", "application_company", "city", "salary_min", "salary_max"]].head(12).copy()
                    preview["salary_range"] = preview.apply(
                        lambda row: format_salary(row["salary_min"], row["salary_max"], row.get("parameters_salary_unit", "")),
                        axis=1,
                    )
                    preview = preview.rename(
                        columns={
                            "title": "Job Title",
                            "application_company": "Employer",
                            "city": "City",
                        }
                    )[["Job Title", "Employer", "City", "salary_range"]]
                    preview = preview.rename(columns={"salary_range": "Salary"})
                    st.dataframe(preview, hide_index=True, use_container_width=True)

with tab6:
    st.header("Usage Insights")
    sqlite_path, csv_path = analytics_artifact_paths(PROJECT_ROOT)
    st.caption(
        "Persistent analytics across sessions from SQLite and CSV logs "
        f"({sqlite_path.name}, {csv_path.name})."
    )

    analytics_events = load_analytics_events(PROJECT_ROOT)
    visit_events = analytics_events[analytics_events["event_type"] == "visit"]
    search_events = analytics_events[analytics_events["event_type"] == "search"]
    recommendation_events = analytics_events[analytics_events["event_type"] == "recommendation"]

    total_searches = len(search_events)
    total_recommendations = len(recommendation_events)

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Visit Count", f"{len(visit_events):,}")
    summary_col2.metric("Search Count", f"{total_searches:,}")
    summary_col3.metric("Recommendations Logged", f"{total_recommendations:,}")

    visit_df = visit_events[["timestamp"]].copy()
    if not visit_df.empty:
        visit_df["timestamp"] = pd.to_datetime(visit_df["timestamp"])
        visit_df["hour"] = visit_df["timestamp"].dt.hour
        visits_by_hour = (
            visit_df.groupby("hour", as_index=True)
            .size()
            .rename("visits")
            .reindex(range(24), fill_value=0)
        )

        st.subheader("Visit Times (Hour of Day)")
        st.bar_chart(visits_by_hour)

    search_df = search_events[["timestamp", "channel", "results_count"]].copy()
    search_df["channel"] = search_df["channel"].replace(
        {
            "job_seeker": "Job Seeker",
            "student_field": "Student Field",
            "veteran": "Veteran",
        }
    )
    if not search_df.empty:
        st.subheader("Search Counter by Workflow")
        st.bar_chart(search_df["channel"].value_counts())

        search_df["timestamp"] = pd.to_datetime(search_df["timestamp"])
        searches_over_time = (
            search_df.set_index("timestamp")
            .sort_index()
            .resample("h")
            .size()
            .rename("searches")
        )
        st.subheader("Search Activity Over Time")
        st.line_chart(searches_over_time)

    recommendation_df = recommendation_events[["title", "city"]].copy()
    if not recommendation_df.empty:
        st.subheader("Top Recommended Roles")
        st.bar_chart(recommendation_df["title"].value_counts().head(10))

        st.subheader("Recommendations by City")
        st.bar_chart(recommendation_df["city"].value_counts().head(10))

st.markdown("---")
with st.expander("About This Tool — Transparency & Limitations"):
    st.markdown(
        """
**Data Source:** National Labor Exchange (NLx) Colorado job postings.

**How matching works:** An NLP pipeline converts unstructured posting text
(title, description, and requirement fields) into structured skill profiles,
then TF-IDF and cosine similarity compare your input against those profiles.

**Integrated analyses:**
- Skill matching and skill-gap extraction
- MOC translation for veteran pathways
- Emerging skill detection (raw vs taxonomy confidence)
- Ghost-job language comparison
- Credential inflation detection by O*NET grouping
- Salary fairness benchmarking by city and role
- Education-to-job alignment from CIP codes

**Compatibility and robustness updates:**
- Uses lowercase resampling alias (`h`) to avoid pandas hourly deprecation warnings.
- Uses escaped O*NET regex (`O\.NET`) for literal-dot matching in taxonomy source filters.
- Uses explicit dataframe column naming after `value_counts().reset_index()`.
- Guards ghost-language vectorization when joined text is empty.
- Normalizes education strings with alias mapping for credential inflation analysis.
- Parses comma-formatted salary text in recruiter quality scoring.
- Escapes MOC codes in direct-match regex search.

**Limitations:**
- Job postings are not hiring outcomes.
- Coverage may be uneven across employers and regions.
- MOC-to-civilian mapping is approximate.
- Salary fields are incomplete in some postings.

This tool is a starting point for exploration, not a final decision system.
"""
    )
