# Streamlit App for Plum Condition Prediction (4 Classes)

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter
# Streamlit App
st.set_page_config(layout="wide", page_title="Plum Classifier")


# Configuration
CLASS_NAMES = ["bruised", "cracked", "rotten", "spotted", "unaffected", "unripe"]
IMAGE_SIZE = (224, 224)
MODEL_PATH = "files/EfficientNetb3-final.keras"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Load model
@st.cache_resource

def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Preprocess function


def preprocess_image(img):
    img = img.convert("RGB")  # <-- Convertir l’image en RGB
    img = img.resize(IMAGE_SIZE)
    img = np.array(img).astype("float32") / 255.0
    return tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

# Classify image

def classify_image(image_tensor):
    preds = model.predict(image_tensor)[0]
    class_index = np.argmax(preds)
    return CLASS_NAMES[class_index], preds[class_index], preds.tolist()

# -----------------------------
# Process Video Frame-by-Frame
# -----------------------------

def process_video(video_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())
    temp_file.close()

    cap = cv2.VideoCapture(temp_file.name)
    predictions = []
    images = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        tensor = preprocess_image(pil_image)
        pred_class, conf, _ = classify_image(tensor)
        predictions.append((pred_class, conf))
        images.append((pil_image, pred_class, conf))

    cap.release()
    return predictions, images
    # Streamlit App for Plum Condition Prediction (4 Classes)


# t-SNE Visualization

def tsne_visualization():
    """
    Loads and displays a precomputed t-SNE plot saved as 't-SNE.png'.
    """
    try:
        st.image("t-SNE.png", caption="t-SNE of Extracted Features", use_column_width=True)
    except FileNotFoundError:
        st.error("t-SNE image not found. Please ensure 'tsne_plot.png' is available.")

def view_metrics():
    """
    Loads and displays a precomputed metrics plot plot saved as 'metrics.png'.
    """
    try:
        st.image("metrics.png", caption="t-SNE of Extracted Features", use_column_width=True)
    except FileNotFoundError:
        st.error("metrics.png image not found. Please ensure metrics.png is available.")

with st.sidebar:
    st.title("Ubunifi AI (Creative AI)")
    show_metrics = st.button("Metrics Insights")
    show_tsne = st.button("t-SNE Plot")

st.title("JCIA 2025 Plum Classification")
st.markdown(
    "<p style='font-size: 1.4em; font-style: italic;color: #4CAF50;;'>"
    "This AI-powered platform allows users to upload plum images or videos to \
    automatically detect fruit quality and categorize them into six classes: bruised, cracked, rotten, spotted, unaffected, and unripe."
    "</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload Image or Video")
    img_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    vid_file = st.file_uploader("Video", type=["mp4", "avi"]) 

    detect_img = detect_vid = False
    if img_file:
        detect_img = st.button("Detect Image")
        st.image(img_file, use_column_width=True)
    elif vid_file:
        detect_vid = st.button("Detect Video")
        st.video(vid_file)

    if detect_img:
        img = Image.open(img_file)
        tensor = preprocess_image(img)
        pred, conf, scores = classify_image(tensor)
        st.session_state.img_result = (img, pred, conf, scores)

    if detect_vid:
        pred_summary, pred_frames = process_video(vid_file)
        st.session_state.vid_result = (pred_summary, pred_frames)

with col2:
    if "img_result" in st.session_state:
        img, pred, conf, scores = st.session_state.img_result
        st.subheader("Image Prediction")
        st.metric("Class", pred)
        st.metric("Confidence", f"{conf:.2%}")
        fig = go.Figure([go.Bar(x=CLASS_NAMES, y=scores)])
        fig.update_layout(title="Class Probabilities")
        st.plotly_chart(fig, use_container_width=True)

    if "vid_result" in st.session_state:
        summary, frames = st.session_state.vid_result
        st.subheader("Video Summary")
        st.write(f"Total Frames: {len(summary)}")
        counts = Counter([p[0] for p in summary])
        confidences = [p[1] for p in summary]
        for cls, count in counts.items():
            st.write(f"{cls}: {count} frames")
        st.metric("Most Common Class", counts.most_common(1)[0][0])
        st.metric("Average Confidence", f"{np.mean(confidences):.2%}")

        fig = go.Figure(go.Indicator(mode="gauge+number", value=np.mean(confidences) * 100, gauge={'axis': {'range': [0, 100]}}))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Frame Preview")
        frame_cols = st.columns(5)
        for i, (im, pr, cf) in enumerate(frames[:10]):
            with frame_cols[i % 5]:
                st.image(im, caption=f"{pr} ({cf:.1%})")

# -------------------------------
# Metrics Display
# -------------------------------

if show_metrics:
    view_metrics()
    
        
# -------------------------------
# t-SNE Display
# -------------------------------

if show_tsne:
    tsne_visualization()
