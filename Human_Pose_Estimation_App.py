import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Constants
DEMO_IMAGE = 'stand.jpg'
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}
POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]
width, height = 368, 368
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Page Configuration
st.set_page_config(page_title="Human Pose Estimation", layout="wide")
st.title("📸 Human Pose Estimation App")
st.markdown("Upload an image to detect human pose key points. Ensure the image is clear and parts are visible!")

# Image Upload
img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    st.warning("No image uploaded. Using the default demo image.")
    image = np.array(Image.open(DEMO_IMAGE))

st.subheader("Original Image")
st.image(image, caption="Uploaded/Demo Image", use_container_width=True)

# Slider for threshold
threshold = st.slider("Detection Threshold", min_value=0, max_value=100, value=20, step=5) / 100

@st.cache_data
def poseDetector(frame, threshold):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])
        points.append((x, y) if conf > threshold else None)
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return frame

# Pose Detection
with st.spinner("Processing..."):
    output_image = poseDetector(image, threshold)

st.subheader("Estimated Pose")
st.image(output_image, caption="Pose Detected", use_container_width=True)

# Footer
st.markdown("""
    #
    
""")
