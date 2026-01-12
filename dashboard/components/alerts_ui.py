import streamlit as st
import time

def render_alerts(history):

    st.subheader("🚨 Alerts")

    if not history["alerts"]:
        st.success("No alerts detected yet")
        return

    last_alert = history["alerts"][-1]

    if last_alert == "CRITICAL":
        st.error("🚨 CRITICAL ALERT DETECTED")
    elif last_alert == "WARNING":
        st.warning("⚠️ WARNING ALERT")
    else:
        st.success("✅ No critical alerts")

    if history["time"]:
        st.caption(
            f"Last alert time: {time.ctime(history['time'][-1])}"
        )
