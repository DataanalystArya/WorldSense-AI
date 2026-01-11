import streamlit as st
import pandas as pd

def render_charts(history):
    st.subheader("📊 Analytics")

    if len(history["time"]) < 3:
        st.info("Not enough data yet")
        return

    df = pd.DataFrame(history)
    df["time"] -= df["time"][0]

    chart = st.selectbox(
        "Chart Type",
        ["Line", "Area", "Bar"],
        key="chart_type"
    )

    if chart == "Line":
        st.line_chart(df.set_index("time"))
    elif chart == "Area":
        st.area_chart(df.set_index("time"))
    else:
        st.bar_chart(df.set_index("time"))
