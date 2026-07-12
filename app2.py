import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Thresholding Techniques", layout="wide")

st.title("Image Segmentation Using Thresholding Techniques")


def show_image(image, caption, channels=None):
    kwargs = {"caption": caption}
    if channels is not None:
        kwargs["channels"] = channels

    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, **kwargs)


uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Global Threshold
    _, global_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Adaptive Mean
    adaptive_mean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Adaptive Gaussian
    adaptive_gaussian = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Otsu Threshold
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    st.subheader("Thresholding Results")

    col1, col2 = st.columns(2)

    with col1:
        show_image(img, "Original Image")
        show_image(global_thresh, "Global Threshold", channels="GRAY")
        show_image(adaptive_mean, "Adaptive Mean", channels="GRAY")

    with col2:
        show_image(gray, "Grayscale", channels="GRAY")
        show_image(adaptive_gaussian, "Adaptive Gaussian", channels="GRAY")
        show_image(otsu, "Otsu Threshold", channels="GRAY")
