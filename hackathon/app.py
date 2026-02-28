import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Colorado Workforce Intelligence",
    page_icon="🏔️",
    layout="wide",
)

from hackathon.core.data import load_data
from hackathon.core.matching import find_matching_jobs
from hackathon.core.student import FIELD_KEYWORDS, top_field_skills, top_skills
from hackathon.core.veterans import MOC_DICTIONARY, veteran_full_match


@st.cache_data
def load_cached_data():
    return load_data()


jobs_clean, skill_profiles, processed = load_cached_data()

st.title("🏔️ Colorado Workforce Intelligence")
st.caption("Powered by NLx data with NLP-extracted skill intelligence")
st.markdown("---")


tab1, tab2, tab3 = st.tabs([
    "💼 Job Seeker",
    "📚 Student / Career Changer",
    "🎖️ Veteran",
])

with tab1:
    st.header("Find Jobs That Match Your Skills")
    st.write("Describe what you can do in plain language.")

    user_skills = st.text_area(
        "Your skills and experience:",
        placeholder=(
            "Example: I have customer service experience, Excel skills, "
            "and enjoy solving problems with people."
        ),
        height=120,
    )

    city_options = ["All Cities"] + sorted([city for city in jobs_clean["city"].unique() if str(city).strip()])
    city_filter = st.selectbox("Filter by city (optional):", city_options)

    if st.button("🔍 Find My Jobs", type="primary"):
        if not user_skills.strip():
            st.warning("Please describe your skills first.")
        else:
            with st.spinner("Matching your skills to Colorado jobs..."):
                results = find_matching_jobs(user_skills, jobs_clean, skill_profiles, top_n=10)
                if city_filter != "All Cities":
                    results = results[results["city"] == city_filter]

            st.success(f"Found {len(results)} matching jobs.")

            for _, row in results.iterrows():
                title = row.get("title", "Untitled Role")
                company = row.get("application_company", "Unknown Company")
                city = row.get("city", "Unknown City")
                with st.expander(f"{title} at {company} — {city}"):
                    score = row.get("match_score", 0)
                    st.metric("Match Score", f"{float(score):.0%}")
                    salary_min = row.get("parameters_salary_min", "")
                    salary_max = row.get("parameters_salary_max", "")
                    if str(salary_min).strip() and str(salary_max).strip():
                        st.write(f"💰 Salary: ${float(salary_min):,.0f} - ${float(salary_max):,.0f}")
                    st.write(f"🎓 Education: {row.get('requirements_min_education', 'Not specified') or 'Not specified'}")
                    st.write(f"⏱️ Experience: {row.get('requirements_experience', 'Not specified') or 'Not specified'}")

with tab2:
    st.header("What Skills Are Trending in Colorado?")
    st.write("See the most demanded skills and career paths.")

    skill_counts = top_skills(processed, limit=20)
    st.subheader("🔥 Top 20 In-Demand Skills")
    st.bar_chart(skill_counts)

    field = st.selectbox("I am interested in:", list(FIELD_KEYWORDS.keys()))

    if st.button("🔍 Show Me This Field", type="primary"):
        with st.spinner("Analyzing this career field..."):
            query = FIELD_KEYWORDS[field]
            results = find_matching_jobs(query, jobs_clean, skill_profiles, top_n=8)
            field_job_ids = results["system_job_id"].tolist()
            field_skill_counts = top_field_skills(processed, field_job_ids, limit=10)

        st.success(f"Top roles and skills for {field}")

        left, right = st.columns(2)
        with left:
            st.subheader("Skills to Learn")
            for skill_name in field_skill_counts.index:
                st.write(f"✅ {skill_name}")

        with right:
            st.subheader("Sample Jobs")
            for _, row in results.head(5).iterrows():
                st.write(f"📌 {row.get('title', 'Untitled')} — {row.get('city', 'Unknown City')}")

with tab3:
    st.header("🎖️ Veteran Career Translator")
    st.write("Translate military occupational experience into civilian Colorado job opportunities.")

    col1, col2 = st.columns([1, 2])

    with col1:
        moc_input = st.text_input("Enter MOS/MOC code:", placeholder="e.g. 68W, 11B, 25B").upper().strip()
        st.markdown("**Common MOS/MOC Codes**")
        for code, (role, _) in list(MOC_DICTIONARY.items())[:8]:
            st.caption(f"{code} — {role}")

    with col2:
        if st.button("🔍 Find My Civilian Career", type="primary"):
            if not moc_input:
                st.warning("Please enter a MOS/MOC code.")
            else:
                with st.spinner("Translating military experience..."):
                    direct_matches, skill_matches, moc_title = veteran_full_match(
                        moc_input,
                        jobs_clean,
                        skill_profiles,
                        top_n=8,
                    )

                st.success(f"Results for {moc_input} — {moc_title}")

                if not direct_matches.empty:
                    st.subheader("⭐ Direct MOC-Tagged Matches")
                    for _, row in direct_matches.head(3).iterrows():
                        st.write(
                            f"📌 {row.get('title', 'Untitled')} at "
                            f"{row.get('application_company', 'Unknown Company')} — "
                            f"{row.get('city', 'Unknown City')}"
                        )

                st.subheader("💡 Skill-Based Matches")
                for _, row in skill_matches.head(6).iterrows():
                    title = row.get("title", "Untitled")
                    city = row.get("city", "Unknown City")
                    score = float(row.get("match_score", 0))
                    with st.expander(f"{title} — {city} ({score:.0%} match)"):
                        st.write(f"🏢 Company: {row.get('application_company', 'Unknown Company')}")
                        st.write(
                            "🎓 Education Required: "
                            f"{row.get('requirements_min_education', 'Not specified') or 'Not specified'}"
                        )

st.markdown("---")
with st.expander("📋 About This Tool — Transparency & Limitations"):
    st.markdown(
        """
**Data Source:** National Labor Exchange (NLx) Colorado job postings.

**How matching works:** An NLP pipeline converts unstructured posting text
(title, description, and requirement fields) into structured skill profiles,
then TF-IDF and cosine similarity compare your input against those profiles.

**Limitations:**
- Job postings are not hiring outcomes.
- Coverage may be uneven across employers and regions.
- MOC-to-civilian mapping is approximate.
- Salary fields are incomplete in some postings.

This tool is a starting point for exploration, not a final decision system.
"""
    )
