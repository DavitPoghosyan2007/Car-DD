
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="AI Car Damage Detector",
    page_icon="🚘",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #101216;
    color: #F5F5F5;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #FFFFFF;
}
div[data-testid="stMetric"] {
    background: #1B1F27;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #303642;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("runs/detect/runs/car_damage/weights/best.pt")

st.title("🚘 AI Car Damage Detector")
st.caption("AI-powered vehicle damage inspection")

@st.cache_resource
def load_model(path):
    return YOLO(str(path))

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH.resolve()}")
    st.info("Check the location of your best.pt file.")
    st.stop()

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png", "webp"]
)

confidence = st.slider(
    "Detection confidence",
    min_value=0.05,
    max_value=0.90,
    value=0.15,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Analyzing the image..."):
        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=640,
            verbose=False
        )

    result = results[0]
    annotated_image = Image.fromarray(result.plot()[..., ::-1])

    left, right = st.columns(2)

    with left:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("AI detection")
        st.image(annotated_image, use_container_width=True)

    boxes = result.boxes
    count = len(boxes)

    st.divider()
    st.subheader("Inspection results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detected damages", count)

    with col2:
        st.metric("Damage categories", len(set(
            int(box.cls[0]) for box in boxes
        )))

    if count == 0:
        st.warning(
            "No damage detected at this confidence threshold. "
            "This does not guarantee that the vehicle is undamaged."
        )
    else:
        rows = []

        for box in boxes:
            class_id = int(box.cls[0])
            score = float(box.conf[0])

            rows.append({
                "Damage type": model.names[class_id],
                "Confidence": f"{score:.1%}"
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

        image_bytes = __import__("io").BytesIO()
        annotated_image.save(image_bytes, format="PNG")

        st.download_button(
            "Download annotated image",
            data=image_bytes.getvalue(),
            file_name="car_damage_analysis.png",
            mime="image/png"
        )

st.divider()
st.caption(
    "AI-assisted inspection only. Results may contain false "
    "detections or missed damage."
)
