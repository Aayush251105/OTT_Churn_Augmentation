"""StayScope entry point: page config, theme CSS, logo and navigation.

Run from the repository root:  streamlit run app/Home.py
"""

from pathlib import Path

import streamlit as st

from components import ui

APP = Path(__file__).resolve().parent
PAGES = APP / "views"  # not "pages/": that folder name triggers Streamlit's legacy auto-navigation

st.set_page_config(page_title="StayScope", page_icon=str(APP / "assets" / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.session_state["_ss_chart_n"] = 0          # chart counter for stable Expand-button keys (components/charts.py)
st.session_state.pop("_ss_chart_title", None)
st.logo(str(APP / "assets" / "logo.svg"), icon_image=str(APP / "assets" / "icon.svg"), size="large")

nav = st.navigation({
    "Overview": [
        st.Page(PAGES / "home.py", title="Home", icon=":material/home:", default=True),
        st.Page(PAGES / "data_explorer.py", title="Data explorer", icon=":material/table_chart:"),
        st.Page(PAGES / "preprocessing.py", title="Preprocessing", icon=":material/cleaning_services:"),
    ],
    "Modelling": [
        st.Page(PAGES / "model_comparison.py", title="Model comparison", icon=":material/compare_arrows:"),
        st.Page(PAGES / "churn_predictor.py", title="Churn predictor", icon=":material/person_search:"),
        st.Page(PAGES / "augmentation_study.py", title="Augmentation study", icon=":material/hub:"),
    ],
    "Insights": [
        st.Page(PAGES / "risk_retention.py", title="Risk & retention", icon=":material/shield:"),
        st.Page(PAGES / "budget_simulator.py", title="Budget simulator", icon=":material/payments:"),
        st.Page(PAGES / "why_customers_leave.py", title="Why customers leave", icon=":material/forum:"),
    ],
})

with st.sidebar:
    st.html('<div class="ss-foot">FDS project · Aayush & Sukhraj</div>')

nav.run()
