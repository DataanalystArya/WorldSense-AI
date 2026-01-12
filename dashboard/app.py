import sys
from pathlib import Path
import time
import numpy as np
import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# ================= PATH FIX =================
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.analytics import Analytics
from utils.scene import infer_scene

from components.sidebar import render_sidebar
from components.charts import render_charts
from components.alerts_ui import render_alerts
from components.summary import render_summary

# ================= CONSTANTS =================
VEHICLE_CLASSES = {
    "car", "bus", "truck", "motorcycle", "motorbike", "bicycle", "van"
}

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="WorldSense-AI",
    page_icon="🌍",
    layout="wide"
)

# ================= SESSION INIT (SAFE) =================
if "history" not in st.session_state:
    st.session_state.history = {}

DEFAULT_HISTORY = {
    "time": [],
    "persons": [],
    "vehicles": [],
    "total": [],
    "scene": [],
    "alerts": []
}

for k, v in DEFAULT_HISTORY.items():
    if k not in st.session_state.history:
        st.session_state.history[k] = v

# ================= MODEL =================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ================= HEADER =================
st.title("🌍 WorldSense-AI")
st.caption("Real-Time Visual Intelligence • Scene Understanding • Alerts")

tabs = st.tabs([
    "🎥 Live Detection",
    "📊 Analytics",
    "🚨 Alerts",
    "🧾 Summary"
])

confidence, uploaded_file, start = render_sidebar()
analytics = Analytics()

# ================= LIVE DETECTION =================
with tabs[0]:

    # ================= VIDEO =================
    if uploaded_file and start and uploaded_file.type == "video/mp4":

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp.name)

        video_col, info_col = st.columns([5, 2])
        frame_box = video_col.empty()
        info_box = info_col.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(
                frame,
                conf=min(confidence, 0.35),
                iou=0.5
            )

            annotated = results[0].plot()

            counts = analytics.count_objects(
                results[0].boxes,
                model.names
            )

            fps = analytics.update_fps()

            persons = counts.get("person", 0)
            vehicles = sum(
                counts.get(cls, 0) for cls in VEHICLE_CLASSES
            )

            scene, context = infer_scene(counts)

            # ================= ALERT LOGIC =================
            alert_level = "NORMAL"
            if persons >= 12:
                alert_level = "CRITICAL"
            elif persons >= 8:
                alert_level = "WARNING"

            if scene in ["Market", "Crowded Street", "Public Gathering"] and persons >= 10:
                alert_level = "CRITICAL"

            # ================= SAVE HISTORY (CORRECT WAY) =================
            st.session_state.history["time"].append(time.time())
            st.session_state.history["persons"].append(persons)
            st.session_state.history["vehicles"].append(vehicles)
            st.session_state.history["total"].append(sum(counts.values()))
            st.session_state.history["scene"].append(scene)
            st.session_state.history["alerts"].append(alert_level)

            frame_box.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

            info_box.markdown(
                f"""
                <div style="padding:15px;border-radius:10px;background:#111;">
                    <h4>FPS</h4>
                    <h2>{fps:.2f}</h2>
                    <hr>
                    <b>Persons:</b> {persons}<br>
                    <b>Vehicles:</b> {vehicles}<br>
                    <b>Scene:</b> {scene}<br>
                    <b>Alert:</b> <span style="color:red;">{alert_level}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        cap.release()
        st.success("✅ Live Video Detection Completed")

    # ================= IMAGE =================
    elif uploaded_file and start and uploaded_file.type.startswith("image"):

        image_bytes = uploaded_file.read()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        results = model(
            image,
            conf=min(confidence, 0.35),
            iou=0.5
        )

        annotated = results[0].plot()

        counts = analytics.count_objects(
            results[0].boxes,
            model.names
        )

        persons = counts.get("person", 0)
        vehicles = sum(
            counts.get(cls, 0) for cls in VEHICLE_CLASSES
        )

        scene, context = infer_scene(counts)

        alert_level = "NORMAL"
        if persons >= 10:
            alert_level = "CRITICAL"
        elif persons >= 6:
            alert_level = "WARNING"

        # ================= SAVE HISTORY =================
        st.session_state.history["time"].append(time.time())
        st.session_state.history["persons"].append(persons)
        st.session_state.history["vehicles"].append(vehicles)
        st.session_state.history["total"].append(sum(counts.values()))
        st.session_state.history["scene"].append(scene)
        st.session_state.history["alerts"].append(alert_level)

        img_col, info_col = st.columns([5, 2])

        img_col.image(
            annotated,
            channels="BGR",
            use_container_width=True
        )

        info_col.markdown(
            f"""
            <div style="padding:15px;border-radius:10px;background:#111;">
                <h4>Total Objects</h4>
                <h2>{sum(counts.values())}</h2>
                <hr>
                <b>Persons:</b> {persons}<br>
                <b>Vehicles:</b> {vehicles}<br>
                <b>Scene:</b> {scene}<br>
                <b>Alert:</b> <span style="color:red;">{alert_level}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("✅ Image Analysis Completed")

    else:
        st.info("⬅ Upload image or video and click Start Analysis")

# ================= ANALYTICS =================
with tabs[1]:
    render_charts(st.session_state.history)

# ================= ALERTS =================
with tabs[2]:
    render_alerts(st.session_state.history)

# ================= SUMMARY =================
with tabs[3]:
    render_summary(st.session_state.history)
