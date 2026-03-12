import streamlit as st

st.set_page_config(
    page_title="AI Asana Analyst",
    page_icon="🧘",
    layout="wide"
)

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Live Mode", "Upload & Report"]
)

if mode == "Live Mode":
    from app.live.live_mode import run_live_mode
    run_live_mode()

elif mode == "Upload & Report":
    from app.report.upload_mode import run_upload_mode
    run_upload_mode()