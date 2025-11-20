# src/face_detection_app.py

import cv2
import numpy as np
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as TF

# -------------------------
# Resolve project root (so this file works inside src/)
# -------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent  # one level up from src/
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "facercnn.pth"
YOLO_FACE_NAME = "yolov8n-face.pt"  # change if you have a specific face model


# -------------------------
# Model loaders (cached)
# -------------------------
@st.cache_resource
def load_yolo_model(model_name=str(YOLO_FACE_NAME)):
    """
    Load YOLOv8 model. If the face model isn't available,
    fall back to generic yolov8n.
    """
    try:
        model = YOLO(model_name)
    except Exception:
        model = YOLO("yolov8n")
    return model


@st.cache_resource
def load_fasterrcnn(checkpoint_path=str(CHECKPOINT_PATH)):
    """
    Load your trained Faster R-CNN model from checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        st.error(f"Could not load Faster R-CNN checkpoint at {checkpoint_path}:\n{e}")
        return None, device
    model.to(device).eval()
    return model, device


# -------------------------
# Prediction helpers
# -------------------------
def predict_yolo_on_image(np_img, yolo_model, conf=0.35):
    """
    Run YOLOv8 on a numpy image (H,W,3 BGR/RGB).
    Returns (N,4) boxes in xyxy format.
    """
    results = yolo_model.predict(np_img, conf=conf, verbose=False)
    if len(results) == 0 or results[0].boxes is None:
        return np.zeros((0, 4))
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes


def predict_frcnn_on_image(pil_img, frcnn_model, device, conf=0.5):
    """
    Run Faster R-CNN on a PIL image.
    Returns boxes, scores filtered by confidence.
    """
    tensor = TF.to_tensor(pil_img).to(device)
    with torch.no_grad():
        out = frcnn_model([tensor])[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    keep = scores >= conf
    return boxes[keep], scores[keep]


def draw_boxes(np_img, boxes, color=(0, 255, 0)):
    """
    Draw rectangles for each box on the image.
    """
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(np_img, (x1, y1), (x2, y2), color, 2)
    return np_img


# -------------------------
# Crowd management helper
# -------------------------
def get_crowd_level(face_count: int) -> str:
    """
    Simple rule-based crowd level.
    Adjust thresholds as you like.
    """
    if face_count == 0:
        return "Empty"
    elif face_count < 5:
        return "Low"
    elif face_count < 15:
        return "Medium"
    else:
        return "High"


def run_detection_on_pil(pil_img, model_choice, confidence):
    """
    Runs the selected model on a PIL image,
    returns (annotated_image_np, face_count, crowd_level).
    """
    np_img = np.array(pil_img)

    if model_choice == "YOLOv8 (Fast)":
        yolo_model = load_yolo_model()
        boxes = predict_yolo_on_image(np_img, yolo_model, conf=confidence)
        color = (0, 255, 0)  # green
    else:
        frcnn_model, device = load_fasterrcnn()
        if frcnn_model is None:
            st.error("Faster R-CNN model not available. Check checkpoint path.")
            return None, 0, "Unknown"
        boxes, scores = predict_frcnn_on_image(pil_img, frcnn_model, device, conf=confidence)
        color = (0, 0, 255)  # red

    face_count = boxes.shape[0]
    level = get_crowd_level(face_count)
    np_img = draw_boxes(np_img, boxes, color=color)
    return np_img, face_count, level


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Crowd Management — Face Detection",
    page_icon="👥",
    layout="centered",
)

st.title("👥 Crowd Management using Face Detection (YOLOv8 + Faster R-CNN)")

st.markdown(
    """
This app detects faces and **estimates crowd level** from webcam snapshots or uploaded images.

- **YOLOv8 (Fast)** → real-time-ish detection, good for larger crowds.  
- **Faster R-CNN (Your Model)** → uses the model you trained on FDDB (slower but custom).  

Crowd level is classified as:  
**Empty / Low / Medium / High** based on the number of detected faces.
"""
)

col1, col2 = st.columns([2, 1])
with col1:
    model_choice = st.selectbox(
        "Select model",
        ["YOLOv8 (Fast)", "Faster R-CNN (Your Model)"],
    )
    confidence = st.slider(
        "Detection confidence threshold", 0.10, 0.95, 0.35, 0.05
    )
    show_count = st.checkbox("Show face count & crowd level", value=True)

with col2:
    st.write("**Info:**")
    st.write("- Webcam snapshot works with both models.")
    st.write("- Uploaded images work with both models.")
    st.write("- Useful for basic crowd estimation.")
    st.write(f"**Checkpoint:** `{CHECKPOINT_PATH.name}`")


# -------------------------
# Webcam snapshot (for crowd management)
# -------------------------
st.subheader("📷 Webcam Snapshot — Crowd Estimation")

cam_img = st.camera_input("Take a picture with your webcam")

if cam_img is not None:
    pil_cam = Image.open(cam_img).convert("RGB")
    result, count, level = run_detection_on_pil(pil_cam, model_choice, confidence)

    if result is not None:
        if show_count:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Estimated People", count)
            with c2:
                st.metric("Crowd Level", level)
        st.image(result, use_column_width=True)

        ok, buf = cv2.imencode(".jpg", result)
        if ok:
            st.download_button(
                "Download webcam result",
                buf.tobytes(),
                file_name="webcam_crowd_detection.jpg",
                mime="image/jpeg",
            )


# -------------------------
# Image upload — crowd management
# -------------------------
st.subheader("📂 Upload Image — Crowd Estimation")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    result, count, level = run_detection_on_pil(pil_img, model_choice, confidence)

    if result is not None:
        if show_count:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Estimated People", count)
            with c2:
                st.metric("Crowd Level", level)
        st.image(result, use_column_width=True)

        ok, buf = cv2.imencode(".jpg", result)
        if ok:
            st.download_button(
                "Download uploaded image result",
                buf.tobytes(),
                file_name="upload_crowd_detection.jpg",
                mime="image/jpeg",
            )
