import streamlit as st
import pandas as pd
import json
import plotly.express as px
from scorer import score_in_batches
from campaign_agent import CampaignAgent
from opportunity_agent import evaluate_single_opportunity
from rag import initialize_rag
from proposal_agent import build_agent as build_proposal_agent
from proposal_pdf import generate_proposal_pdf

st.set_page_config(
    page_title="Saudi Companies Directory",
    page_icon="🇸🇦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; }

.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00c853, #00e676);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.header-sub {
    color: #666;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00c853; }
.metric-num { font-size: 2.2rem; font-weight: 800; color: #00c853; }
.metric-lbl { font-size: 0.8rem; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

.detail-box {
    background: #1a1d27;
    border: 1px solid #00c853;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}
.detail-title { font-size: 1.4rem; font-weight: 800; color: #fff; margin-bottom: 6px; }
.detail-desc { color: #aaa; line-height: 1.7; margin: 12px 0; }
.detail-row { display: flex; gap: 30px; flex-wrap: wrap; margin-top: 16px; }
.detail-key { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
.detail-val { font-size: 0.95rem; color: #eee; font-weight: 600; margin-top: 2px; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-sector { background: #1e3a5f; color: #60a5fa; }
.badge-startup { background: #1a3a2a; color: #4ade80; }
.badge-emp { background: #2a1f3a; color: #c084fc; }

.filter-header {
    color: #00c853;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
    margin-top: 16px;
}

div[data-testid="stSidebar"] {
    background: #12151e;
    border-right: 1px solid #1e2130;
}

.no-results {
    text-align: center;
    padding: 60px 20px;
    color: #555;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    with open('companies.json', encoding='utf-8') as f:
        return pd.DataFrame(json.load(f))

df = load_data()

# ── Header ─────────────────────────────────────────────────────────
st.markdown('<div class="header-title">🇸🇦 Saudi Companies Directory</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Browse & filter companies across all sectors — click any company for full details</div>', unsafe_allow_html=True)

# ── Sidebar Filters ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Filter Companies")

    st.markdown('<div class="filter-header">📂 Sector — pick one or more</div>', unsafe_allow_html=True)
    all_sectors = sorted(df['sector'].dropna().unique())
    selected_sectors = st.multiselect("", all_sectors, default=[], key="sectors", label_visibility="collapsed", placeholder="All sectors (default)")

    st.markdown('<div class="filter-header">📍 City — pick one or more</div>', unsafe_allow_html=True)
    all_cities = sorted([c for c in df['city_clean'].dropna().unique() if c not in ('Unknown', '')])
    selected_cities = st.multiselect("", all_cities, key="cities", label_visibility="collapsed", placeholder="All cities (default)")

    st.markdown('<div class="filter-header">👥 Company Size — pick one or more</div>', unsafe_allow_html=True)
    emp_order = ['51-200', '200-1,000', '1,000-5,000', '5,000-10,000', '10,000+']
    selected_emp = st.multiselect("", emp_order, default=[], key="emp", label_visibility="collapsed", placeholder="All sizes (default)")

    st.markdown('<div class="filter-header">🚀 Company Type</div>', unsafe_allow_html=True)
    company_type = st.radio("", ['All', 'Startups Only', 'Established Only'], key="type", label_visibility="collapsed")

    st.markdown('<div class="filter-header">🔎 Search</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Company ", key="search", label_visibility="collapsed")

# ── Apply Filters ────────────────────────────────────────────────────
filtered = df.copy()

if selected_sectors:  # empty = show all
    filtered = filtered[filtered['sector'].isin(selected_sectors)]

if selected_cities:  # empty = show all
    filtered = filtered[filtered['city_clean'].isin(selected_cities)]

if selected_emp:  # empty = show all
    filtered = filtered[filtered['emp_bucket'].isin(selected_emp)]

if company_type == 'Startups Only':
    filtered = filtered[filtered['is_startup'] == True]
elif company_type == 'Established Only':
    filtered = filtered[filtered['is_startup'] == False]

if search:
    mask = (
        filtered['name'].str.contains(search, case=False, na=False) |
        filtered['description'].str.contains(search, case=False, na=False) |
        filtered['sub_sector'].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values('name').reset_index(drop=True)

# ── Metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, len(filtered), "Companies Found"),
    (c2, filtered['sector'].nunique(), "Sectors"),
    (c3, int(filtered['is_startup'].sum()), "🚀 Startups"),
    (c4, filtered[filtered['city_clean'] != 'Unknown']['city_clean'].nunique(), "Cities"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-num">{num}</div><div class="metric-lbl">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── State ────────────────────────────────────────────────────────────
if 'selected' not in st.session_state:
    st.session_state.selected = None
if 'score_results' not in st.session_state:
    st.session_state.score_results = None
if 'score_criteria' not in st.session_state:
    st.session_state.score_criteria = ""
if 'scored_count' not in st.session_state:
    st.session_state.scored_count = 0
if 'campaign_emails' not in st.session_state:
    st.session_state.campaign_emails = None
if 'opportunity_results' not in st.session_state:
    st.session_state.opportunity_results = {}
if 'proposal_pdf_path' not in st.session_state:
    st.session_state.proposal_pdf_path = None
if 'proposal_text' not in st.session_state:
    st.session_state.proposal_text = None

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📋 Companies List", "📊 Analytics", "🎯 Lead Scoring", "📧 Campaign", "🤝 Opportunity", "📄 Proposal"])

with tab1:
    if filtered.empty:
        st.markdown('<div class="no-results">😕 No companies match your current filters.</div>', unsafe_allow_html=True)
    else:
        # Detail panel
        if st.session_state.selected is not None:
            idx = st.session_state.selected
            if idx < len(filtered):
                row = filtered.iloc[idx]
                website = str(row.get('website', '') or '').strip()
                website_clean = website.replace('www.', '').strip()

                with st.container(border=True):
                    title_col, close_col = st.columns([5, 1])
                    with title_col:
                        st.markdown(f"### 🏢 {row.get('name', '')}")
                    with close_col:
                        if st.button("✕ Close", key="close"):
                            st.session_state.selected = None
                            st.rerun()
                    badges = f"`{row.get('sector', '')}`"
                    if row.get('is_startup'):
                        badges += "  `🚀 Startup`"
                    badges += f"  `👥 {row.get('employees', 'N/A')}`"
                    st.markdown(badges)
                    st.markdown(f"> {row.get('description', 'No description available.')}")
                    d1, d2, d3, d4 = st.columns(4)
                    with d1:
                        st.markdown("**📍 City**")
                        st.write(row.get('city_clean', 'N/A'))
                    with d2:
                        st.markdown("**🏷️ Sub-sector**")
                        st.write(row.get('sub_sector', 'N/A'))
                    with d3:
                        st.markdown("**👥 Employees**")
                        st.write(row.get('employees', 'N/A'))
                    with d4:
                        st.markdown("**🔗 Website**")
                        if website_clean and website_clean not in ('N/A', 'nan', ''):
                            st.markdown(f"[{website_clean}](https://{website_clean})")
                        else:
                            st.write("N/A")

                st.markdown("---")

        # Grid
        cols_per_row = 2
        rows = [filtered.iloc[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]

        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col_idx, (_, company) in enumerate(row_df.iterrows()):
                abs_idx = filtered.index.get_loc(company.name)
                with cols[col_idx]:
                    startup_tag = "🚀 " if company.get('is_startup') else ""
                    sub = str(company.get('sub_sector', '') or '')[:45]
                    btn_label = f"{startup_tag}**{company['name']}**\n📍 {company.get('city_clean','?')}  ·  👥 {company.get('emp_bucket','?')}\n🏷️ {sub}"
                    if st.button(btn_label, key=f"btn_{abs_idx}", use_container_width=True):
                        st.session_state.selected = abs_idx
                        st.rerun()

with tab2:
    if filtered.empty:
        st.warning("No data to display.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            sector_counts = filtered['sector'].value_counts().reset_index()
            sector_counts.columns = ['Sector', 'Count']
            fig1 = px.bar(sector_counts, x='Count', y='Sector', orientation='h',
                         title='Companies by Sector', color='Count',
                         color_continuous_scale=['#1a3a2a', '#00c853'],
                         template='plotly_dark')
            fig1.update_layout(showlegend=False, height=380, plot_bgcolor='#1a1d27', paper_bgcolor='#1a1d27')
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            city_df = filtered[filtered['city_clean'].notna() & (filtered['city_clean'] != 'Unknown')]
            if not city_df.empty:
                city_counts = city_df['city_clean'].value_counts().reset_index()
                city_counts.columns = ['City', 'Count']
                fig2 = px.pie(city_counts, values='Count', names='City',
                             title='Distribution by City', hole=0.45,
                             template='plotly_dark',
                             color_discrete_sequence=px.colors.qualitative.Safe)
                fig2.update_layout(height=380, paper_bgcolor='#1a1d27')
                st.plotly_chart(fig2, use_container_width=True)

        emp_counts = filtered[
            filtered['emp_bucket'].notna() & (filtered['emp_bucket'] != 'Unknown')
        ]['emp_bucket'].value_counts().reset_index()
        emp_counts.columns = ['Size', 'Count']
        emp_order_f = [e for e in emp_order if e in emp_counts['Size'].values]
        emp_counts['Size'] = pd.Categorical(emp_counts['Size'], categories=emp_order_f, ordered=True)
        emp_counts = emp_counts.sort_values('Size')
        fig3 = px.bar(emp_counts, x='Size', y='Count', title='Companies by Employee Count',
                     color='Count', color_continuous_scale=['#1e3a5f', '#60a5fa'],
                     template='plotly_dark')
        fig3.update_layout(showlegend=False, plot_bgcolor='#1a1d27', paper_bgcolor='#1a1d27')
        st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.markdown("### 🎯 Lead Scoring")
    st.markdown(f"Scoring will run on the **{len(filtered)} companies** from your current filters.")

    if filtered.empty:
        st.warning("No companies to score. Adjust your filters first.")
    else:
        st.markdown("---")

        # Criteria selection
        use_defaults = st.checkbox(
            "✅ Use BeamData default criteria (IT, Fintech, Telecom, Healthcare — 200+ employees — Riyadh priority)",
            value=True
        )

        custom_criteria = ""
        if not use_defaults:
            custom_criteria = st.text_area(
                "✍️ Write your own criteria:",
                placeholder="e.g. I want companies in healthcare with 500+ employees that are likely to invest in AI automation...",
                height=120
            )

        st.markdown("---")

        # Agent mode toggle
        use_agent = st.toggle(
            "🤖 Agent Mode — searches the web for each company (slower but more accurate)",
            value=False
        )
        if use_agent:
            st.info(f"⚠️ Agent mode will do {len(filtered)} web searches — recommended max 10 companies.")

        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            score_btn = st.button(
                f"🎯 Score {len(filtered)} Companies",
                use_container_width=True,
                type="primary"
            )
        with col_info:
            if use_agent:
                st.caption("🤖 Agent will search the web for each company then score it.")
            else:
                st.caption("⚡ Fast mode: scores based on existing data.")

        if score_btn:
            if not use_defaults and not custom_criteria.strip():
                st.error("Please write your criteria or use BeamData defaults.")
            else:
                companies_list = filtered.to_dict(orient='records')

                if use_agent:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(current, total, company_name):
                        progress_bar.progress(current / total)
                        status_text.text(f"🔍 Researching {current}/{total}: {company_name}")

                    try:
                        results = score_in_batches(
                            companies_list,
                            criteria=custom_criteria,
                            use_beamdata_defaults=use_defaults,
                            use_agent=True,
                            progress_callback=update_progress
                        )
                        progress_bar.progress(1.0)
                        status_text.text("✅ Done!")
                        st.session_state.score_results = results
                        st.session_state.scored_count = len(results)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    with st.spinner(f"⚡ Scoring {len(companies_list)} companies..."):
                        try:
                            results = score_in_batches(
                                companies_list,
                                criteria=custom_criteria,
                                use_beamdata_defaults=use_defaults,
                                batch_size=15,
                                use_agent=False
                            )
                            st.session_state.score_results = results
                            st.session_state.scored_count = len(results)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        # Show results
        if st.session_state.score_results:
            results = st.session_state.score_results

            # Summary metrics
            high = sum(1 for r in results if r.get('grade') == 'High')
            medium = sum(1 for r in results if r.get('grade') == 'Medium')
            low = sum(1 for r in results if r.get('grade') == 'Low')

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-num">{len(results)}</div><div class="metric-lbl">Scored</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4ade80">{high}</div><div class="metric-lbl">🟢 High</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#facc15">{medium}</div><div class="metric-lbl">🟡 Medium</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#f87171">{low}</div><div class="metric-lbl">🔴 Low</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Grade filter
            grade_filter = st.radio("Show:", ["All", "🟢 High", "🟡 Medium", "🔴 Low"], horizontal=True)

            filtered_results = results
            if grade_filter == "🟢 High":
                filtered_results = [r for r in results if r.get('grade') == 'High']
            elif grade_filter == "🟡 Medium":
                filtered_results = [r for r in results if r.get('grade') == 'Medium']
            elif grade_filter == "🔴 Low":
                filtered_results = [r for r in results if r.get('grade') == 'Low']

            st.markdown("---")

            # Results table
            for i, r in enumerate(filtered_results):
                grade = r.get('grade', '')
                score = r.get('score', 0)
                emoji = "🟢" if grade == "High" else "🟡" if grade == "Medium" else "🔴"

                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.markdown(f"**{i+1}. {r.get('name', 'N/A')}**")
                        st.caption(f"📍 {r.get('city_clean', 'N/A')}  ·  🏷️ {r.get('sector', 'N/A')}  ·  👥 {r.get('employees', 'N/A')}")
                    with c2:
                        st.markdown(f"### {emoji} {score}")
                    with c3:
                        st.markdown(f"`{grade}`")
                    st.markdown(f"💬 *{r.get('reason', '')}*")
                    if r.get('research'):
                        with st.expander("🔍 Web Research"):
                            st.caption(r.get('research', ''))

            st.markdown("---")

            # Export CSV
            import csv, io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=['name', 'city_clean', 'sector', 'employees', 'score', 'grade', 'reason', 'website'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'name': r.get('name', ''),
                    'city_clean': r.get('city_clean', ''),
                    'sector': r.get('sector', ''),
                    'employees': r.get('employees', ''),
                    'score': r.get('score', ''),
                    'grade': r.get('grade', ''),
                    'reason': r.get('reason', ''),
                    'website': r.get('website', ''),
                })
            csv_data = output.getvalue()

            st.download_button(
                label="⬇️ Export Results as CSV",
                data=csv_data,
                file_name="lead_scoring_results.csv",
                mime="text/csv",
                use_container_width=True
            )

            if st.button("🔄 Clear Results & Re-score", use_container_width=True):
                st.session_state.score_results = None
                st.rerun()

with tab4:
    st.markdown("### 📧 Campaign — Generate Cold Emails")

    if not st.session_state.score_results:
        st.warning("⚠️ Run Lead Scoring first (in the 🎯 Lead Scoring tab), then come back here.")
    else:
        st.markdown("Pick the companies you want to send a personalized email to:")

        results = st.session_state.score_results

        # ── Checkbox picker ─────────────────────────────────────────
        selected_names = []
        for i, r in enumerate(results):
            grade = r.get('grade', '')
            emoji = "🟢" if grade == "High" else "🟡" if grade == "Medium" else "🔴"
            label = f"{emoji} {r.get('name', 'N/A')} — {r.get('sector', 'N/A')} ({r.get('score', 0)})"
            checked = st.checkbox(label, key=f"campaign_pick_{i}")
            if checked:
                selected_names.append(i)

        st.markdown("---")
        st.caption(f"✅ Selected: {len(selected_names)} companies")

        generate_btn = st.button(
            f"📧 Generate Emails for {len(selected_names)} Selected Companies",
            type="primary",
            disabled=(len(selected_names) == 0)
        )

        if generate_btn:
            companies_to_email = [results[i] for i in selected_names]

            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_campaign_progress(current, total, name):
                progress_bar.progress(current / total)
                status_text.text(f"✍️ Writing email {current}/{total}: {name}")

            try:
                agent = CampaignAgent()
                emails = agent.run_on_companies(
                    companies_to_email,
                    progress_callback=update_campaign_progress
                )
                progress_bar.progress(1.0)
                status_text.text("✅ Done!")
                st.session_state.campaign_emails = emails
                st.rerun()
            except Exception as e:
                st.error(f"Error generating emails: {e}")

        # ── Show generated emails ───────────────────────────────────
        if st.session_state.campaign_emails:
            emails = st.session_state.campaign_emails
            st.markdown("---")
            st.markdown(f"#### ✉️ {len(emails)} Emails Generated")

            for i, e in enumerate(emails):
                with st.container(border=True):
                    st.markdown(f"**{i+1}. {e.get('company_name', 'N/A')}**")
                    st.caption(f"📍 {e.get('city', 'N/A')}  ·  🏷️ {e.get('sector', 'N/A')}")
                    st.markdown(f"**Subject:** {e.get('email_subject', '')}")
                    st.text_area(
                        "Body",
                        value=e.get('email_body', ''),
                        height=160,
                        key=f"email_body_{i}",
                        label_visibility="collapsed"
                    )
                    st.caption(f"🎯 Goal: {e.get('campaign_goal', '')}  ·  💡 Suggested: {e.get('suggested_service', '')}")

            st.markdown("---")

            # Export CSV
            import csv as csv_module, io as io_module
            output = io_module.StringIO()
            fieldnames = [
                "company_name", "arabic_name", "sector", "sub_sector", "city",
                "country", "website", "email", "phone", "linkedin_url",
                "employees", "founded_year", "description", "tags",
                "email_subject", "email_body", "campaign_goal", "suggested_service"
            ]
            writer = csv_module.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for e in emails:
                writer.writerow(e)
            csv_data = output.getvalue()

            st.download_button(
                label="⬇️ Export Emails as CSV",
                data=csv_data,
                file_name="campaign_emails.csv",
                mime="text/csv",
                use_container_width=True
            )

            if st.button("🔄 Clear Emails", use_container_width=True):
                st.session_state.campaign_emails = None
                st.rerun()

with tab5:
    st.markdown("### 🤝 Opportunity Qualification")
    st.markdown(
        "After you send a campaign email and the company replies "
        "(by email or call), enter what they said here. The AI will "
        "score the opportunity and tell you if it's ready for a proposal."
    )

    if not st.session_state.campaign_emails:
        st.warning("⚠️ Generate campaign emails first (in the 📧 Campaign tab), then come back here.")
    else:
        emails = st.session_state.campaign_emails
        st.markdown("---")

        company_options = [e.get("company_name", f"Company {i}") for i, e in enumerate(emails)]
        picked_name = st.selectbox("Select a company to log their response:", company_options)
        picked_idx = company_options.index(picked_name)
        picked_email = emails[picked_idx]

        with st.container(border=True):
            st.markdown(f"**📨 Email sent to: {picked_email.get('company_name', 'N/A')}**")
            st.caption(f"Subject: {picked_email.get('email_subject', '')}")

            col1, col2 = st.columns(2)
            with col1:
                budget = st.selectbox("💰 Budget", ["High", "Medium", "Low", "Unknown"], key="opp_budget")
                decision_maker = st.selectbox("🧑‍💼 Decision Maker", ["Available", "Partially available", "Not available"], key="opp_dm")
                timeline = st.selectbox("⏱️ Timeline", ["<3 months", "3-6 months", "Unknown"], key="opp_timeline")
            with col2:
                agreement_level = st.selectbox("🤝 Agreement Level", ["Strong", "Medium", "Weak"], key="opp_agreement")
                potential_value = st.selectbox("📈 Potential Value", ["High", "Medium", "Low"], key="opp_value")
                industry = st.text_input("🏷️ Industry", value=picked_email.get("sector", ""), key="opp_industry")

            need = st.text_area("🎯 What do they need? (from their reply)", key="opp_need", height=80)
            email_response = st.text_area("✉️ Their reply (paste it or summarize it)", key="opp_response", height=100)

            evaluate_btn = st.button("🎯 Evaluate This Opportunity", type="primary", use_container_width=True)

            if evaluate_btn:
                if not email_response.strip():
                    st.error("Please enter their reply before evaluating.")
                else:
                    opp_data = {
                        "company_name": picked_email.get("company_name", ""),
                        "industry": industry,
                        "budget": budget,
                        "email_response": email_response,
                        "need": need,
                        "decision_maker": decision_maker,
                        "timeline": timeline,
                        "agreement_level": agreement_level,
                        "potential_value": potential_value,
                    }
                    with st.spinner("🤖 Evaluating opportunity..."):
                        try:
                            result = evaluate_single_opportunity(opp_data)
                            opp_data.update(result)
                            st.session_state.opportunity_results[picked_email.get("company_name", "")] = opp_data
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        # ── Show result for selected company, if evaluated ─────────
        existing = st.session_state.opportunity_results.get(picked_email.get("company_name", ""))
        if existing:
            st.markdown("---")
            score = existing.get("opportunity_score", 0)
            status = existing.get("status", "")
            emoji = "🟢" if status == "Qualified" else "🔴" if status == "Not Qualified" else "⚪"
            with st.container(border=True):
                st.markdown(f"### {emoji} Score: {score}/100 — `{status}`")
                st.markdown(f"💬 **Reason:** {existing.get('reason', '')}")
                st.markdown(f"➡️ **Next step:** {existing.get('recommended_next_step', '')}")

        # ── Summary table of all evaluated opportunities ───────────
        if st.session_state.opportunity_results:
            st.markdown("---")
            st.markdown("#### 📊 All Evaluated Opportunities")

            all_results = list(st.session_state.opportunity_results.values())
            qualified = sum(1 for r in all_results if r.get("status") == "Qualified")

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-num">{len(all_results)}</div><div class="metric-lbl">Evaluated</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4ade80">{qualified}</div><div class="metric-lbl">🟢 Qualified — Proposal Ready</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            for r in sorted(all_results, key=lambda x: x.get("opportunity_score", 0), reverse=True):
                status = r.get("status", "")
                emoji = "🟢" if status == "Qualified" else "🔴" if status == "Not Qualified" else "⚪"
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{r.get('company_name', 'N/A')}**")
                        st.caption(f"💬 {r.get('reason', '')}")
                    with c2:
                        st.markdown(f"### {emoji} {r.get('opportunity_score', 0)}")

            st.markdown("---")

            import csv as csv_module2, io as io_module2
            output2 = io_module2.StringIO()
            fieldnames2 = [
                "company_name", "industry", "budget", "email_response", "need",
                "decision_maker", "timeline", "agreement_level", "potential_value",
                "opportunity_score", "status", "reason", "recommended_next_step"
            ]
            writer2 = csv_module2.DictWriter(output2, fieldnames=fieldnames2, extrasaction='ignore')
            writer2.writeheader()
            for r in all_results:
                writer2.writerow(r)
            csv_data2 = output2.getvalue()

            st.download_button(
                label="⬇️ Export Opportunities as CSV",
                data=csv_data2,
                file_name="qualified_opportunities.csv",
                mime="text/csv",
                use_container_width=True
            )

with tab6:
    st.markdown("### 📄 Proposal Generator")
    st.markdown(
        "Pick a **Qualified** opportunity, confirm the agreed price, and the "
        "agent will research the company online and in Beam Data's knowledge "
        "base to write a tailored proposal — exported as a PDF."
    )

    qualified = {
        name: r for name, r in st.session_state.opportunity_results.items()
        if r.get("status") == "Qualified"
    }

    if not qualified:
        st.warning("⚠️ No qualified opportunities yet. Evaluate one in the 🤝 Opportunity tab first.")
    else:
        st.markdown("---")
        company_options2 = list(qualified.keys())
        picked_company = st.selectbox("Select a qualified company:", company_options2, key="proposal_company_pick")
        picked_opp = qualified[picked_company]

        with st.container(border=True):
            st.markdown(f"**🏢 {picked_company}**")
            st.caption(f"💬 {picked_opp.get('reason', '')}")

            agreed_price = st.text_input(
                "💰 Agreed price (optional — leave blank to let the agent estimate)",
                placeholder="e.g. $15,000 for a 3-month engagement",
                key="proposal_price_input"
            )

            generate_proposal_btn = st.button("📄 Generate Proposal", type="primary", use_container_width=True)

        if generate_proposal_btn:
            try:
                with st.spinner("Building knowledge base (first run may take a minute)…"):
                    retriever = initialize_rag()
                    proposal_executor = build_proposal_agent(retriever)

                status_ph = st.empty()
                full_output = ""

                with st.spinner(f"Researching **{picked_company}** and drafting the proposal…"):
                    for event in proposal_executor.stream({
                        "company_name": picked_company,
                        "agreed_price": agreed_price.strip() if agreed_price.strip() else "Not specified — please estimate",
                    }):
                        if "actions" in event:
                            for action in event["actions"]:
                                tool_label = "🌐 Searching web" if "web" in action.tool else "📚 Querying knowledge base"
                                status_ph.markdown(f"🔄 {tool_label}: *{str(action.tool_input)[:80]}*")
                        if "output" in event:
                            full_output = event["output"]

                status_ph.empty()

                if full_output:
                    st.session_state.proposal_text = full_output
                    pdf_path = f"/tmp/proposal_{picked_company.replace(' ', '_')}.pdf"
                    generate_proposal_pdf(picked_company, full_output, pdf_path)
                    st.session_state.proposal_pdf_path = pdf_path
                    st.rerun()
                else:
                    st.error("Something went wrong generating the proposal. Check your API key and try again.")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state.proposal_text:
            st.markdown("---")
            st.markdown(f"#### ✅ Proposal Ready")
            with st.container(border=True):
                st.markdown(st.session_state.proposal_text)

            if st.session_state.proposal_pdf_path:
                with open(st.session_state.proposal_pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Proposal (PDF)",
                        data=f.read(),
                        file_name=st.session_state.proposal_pdf_path.split("/")[-1],
                        mime="application/pdf",
                        use_container_width=True
                    )

            if st.button("🔄 Clear Proposal", use_container_width=True):
                st.session_state.proposal_text = None
                st.session_state.proposal_pdf_path = None
                st.rerun()

st.markdown("---")
st.caption("Saudi Companies Directory • Built with Streamlit & Plotly")