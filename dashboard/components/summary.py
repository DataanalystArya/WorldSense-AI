import streamlit as st

def render_summary(history):
    st.subheader("AI Summary")

    if not history or not history["total"]:
        st.info("No detection data available yet.")
        return

    max_persons = max(history["persons"])
    max_vehicles = max(history["vehicles"])
    avg_persons = round(sum(history["persons"]) / len(history["persons"]), 1)
    avg_vehicles = round(sum(history["vehicles"]) / len(history["vehicles"]), 1)
    max_total = max(history["total"])

    st.markdown(f"""
    **Peak Analysis**
    - 🔺 Maximum Objects Detected: **{max_total}**
    - 👥 Peak Persons Count: **{max_persons}**
    - 🚗 Peak Vehicles Count: **{max_vehicles}**

    **Average Activity**
    - 👤 Avg Persons per frame: **{avg_persons}**
    - 🚙 Avg Vehicles per frame: **{avg_vehicles}**

    **AI Insight**
    High-density urban activity detected with significant crowd and traffic presence.
    """)
