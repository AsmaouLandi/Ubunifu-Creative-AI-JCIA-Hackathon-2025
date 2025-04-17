# Streamlit App for Plum Condition Prediction (6 Classes)

import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd 
import tempfile
import tensorflow as tf
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import Counter

# Streamlit App Settings
st.set_page_config(layout="wide", page_title="Plum Classifier")

# Configuration
CLASS_NAMES = ["bruised", "cracked", "rotten", "spotted", "unaffected", "unripe"]
IMAGE_SIZE = (224, 224)
MODEL_PATH = "files/EfficientNetb3-final.keras"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Load model with caching
@st.cache_resource

def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Image preprocessing function
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img = np.array(img).astype("float32") / 255.0
    return tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

# Classify image and return top prediction and probabilities
def classify_image(image_tensor):
    preds = model.predict(image_tensor)[0]
    class_index = np.argmax(preds)
    return CLASS_NAMES[class_index], preds[class_index], preds.tolist()

# Process each frame in video and classify it
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

# Show t-SNE image if available
def tsne_visualization():
    try:
        st.image("t-SNE.png", caption="t-SNE of Extracted Features", use_column_width=True)
    except FileNotFoundError:
        st.error("t-SNE image not found. Please ensure 't-SNE.png' is available.")

# Show precomputed metrics image if available
def view_metrics():
    try:
        st.image("metrics.png", caption="Model Evaluation Metrics", use_column_width=True)
    except FileNotFoundError:
        st.error("metrics.png image not found. Please ensure metrics.png is available.")

# Sidebar for additional plots
with st.sidebar:
    st.title("Ubunifu (Creative) AI")
    show_metrics = st.button("Model Metrics")
    show_tsne = st.button("t-SNE Plot")

# Main App Title
st.title("JCIA 2025 Plum Classification")
st.markdown(
    """
    <p style='font-size: 1.4em; font-style: italic;color: #4CAF50;'>
    This AI-powered platform allows users to upload plum images or videos to \
    automatically detect fruit quality and categorize them into six classes: bruised, cracked, rotten, spotted, unaffected, and unripe.
    </p>
    """,
    unsafe_allow_html=True
)

# Layout for uploading and result viewing
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload Image or Video")
    img_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    vid_file = st.file_uploader("Video", type=["mp4", "avi"])

    # Clear results when file is removed manually by user
    if "img_result" in st.session_state and img_file is None:
        del st.session_state.img_result
        st.rerun()

    if "vid_result" in st.session_state and vid_file is None:
        del st.session_state.vid_result
        st.rerun()

    detect_img = detect_vid = False
    if img_file:
        detect_img = st.button("Detect Image")
        st.image(img_file, use_column_width=True)
    elif vid_file:
        detect_vid = st.button("Detect Video")
        with st.container():
            st.markdown("### Preview of Uploaded Video")
            video_col, _ = st.columns([1, 4])
            with video_col:
                st.video(vid_file)

    # Image prediction logic
    if detect_img:
        img = Image.open(img_file)
        tensor = preprocess_image(img)
        pred, conf, scores = classify_image(tensor)
        st.session_state.img_result = (img, pred, conf, scores)

        df_img = pd.DataFrame([{
            "Image Name": img_file.name,
            "Predicted Class": pred,
            "Confidence": conf,
            **{f"Prob_{cls}": prob for cls, prob in zip(CLASS_NAMES, scores)}
        }])
        df_img.to_csv("image_prediction_result.csv", index=False)
        st.success("Image prediction saved to image_prediction_result.csv")

    # Video prediction logic
    if detect_vid:
        with st.spinner("Processing video... Please wait."):
            video_bytes = vid_file.read()
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(video_bytes)
            temp_file.close()

            cap = cv2.VideoCapture(temp_file.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            predictions = []
            images = []
            progress_bar = st.progress(0)
            frame_count = 0

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

                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()

            df_vid = pd.DataFrame([{
                "Frame": idx + 1,
                "Predicted Class": cls,
                "Confidence": conf
            } for idx, (cls, conf) in enumerate(predictions)])
            df_vid.to_csv("video_prediction_results.csv", index=False)
            st.success("Video prediction results saved to video_prediction_results.csv")

            st.session_state.vid_result = (predictions, images)

# Display prediction results in second column
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

# Optionally display metrics or t-SNE visualization
if show_metrics:
    view_metrics()

if show_tsne:
    tsne_visualization()
