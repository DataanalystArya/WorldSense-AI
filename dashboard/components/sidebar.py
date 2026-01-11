import streamlit as st

def render_sidebar():
    st.sidebar.header("🎛 Control Panel")

    confidence = st.sidebar.slider(
        "Detection Confidence", 0.2, 1.0, 0.5
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Image / Video (upto 2GB)",
        ["mp4", "jpg", "png", "jpeg"]
    )

    start = st.sidebar.button("▶ Start Analysis")

    return confidence, uploaded_file, start
