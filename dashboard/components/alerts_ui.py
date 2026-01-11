import streamlit as st
import time

def render_alerts(history):
    st.subheader("🚨 Alerts")

    if not history or len(history.get("time", [])) == 0:
        st.info("No alerts detected yet.")
        return

    # Last timestamp
    last_alert_time = history["time"][-1]

    persons = history["persons"][-1]
    vehicles = history["vehicles"][-1]

    # ---- SIMPLE ALERT LOGIC ----
    alerts = []

    if persons >= 5:
        alerts.append("👥 Crowd detected")

    if vehicles >= 3:
        alerts.append("🚗 High vehicle density")

    if not alerts:
        st.success("✅ No critical alerts")
    else:
        for alert in alerts:
            st.warning(alert)

    st.caption(f"Last alert time: {time.ctime(last_alert_time)}")
