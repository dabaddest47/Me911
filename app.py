import streamlit as st
import cv2
import numpy as np
from utils import enhance_image_and_extract_text

st.title("Hidden Text Revealer (Blue Shading OCR)")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

    with st.spinner("Processing image to reveal hidden text..."):
        extracted_text, processed_image = enhance_image_and_extract_text(image)

    st.image(processed_image, channels="GRAY", caption="Processed Image", use_column_width=True)
    st.subheader("Extracted Text")
    st.text(extracted_text)