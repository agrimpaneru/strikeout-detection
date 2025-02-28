import streamlit as st
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import numpy as np
import os

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trocr_ocr import TrOCROCR
from utils.image_processing import preprocess_image
from utils.box_sorting import sort_boxes

# Define page configuration
st.set_page_config(
    page_title="Medical Form Text Recognition",
    page_icon="📝",
    layout="wide"
)

# Add a title and description
st.title("Medical Form OCR Pipeline")
st.markdown("""
This application detects and processes text from medical forms:
1. Detects text regions using YOLO
2. Classifies each region as normal text or strike-through
3. Performs OCR on normal text regions
""")

# Define the device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
st.sidebar.info(f"Using device: {device}")

class OCRPipeline:
    def __init__(self):
        # Display loading message while initializing models
        with st.sidebar.expander("Model Loading Status", expanded=True):
            yolo_loading = st.empty()
            yolo_loading.text("Loading YOLO detector...")
            self.yolo_detector = YOLODetector()
            yolo_loading.text("✅ YOLO detector loaded")
            
            resnet_loading = st.empty()
            resnet_loading.text("Loading ResNet classifier...")
            self.resnet_classifier = ResNetClassifier(device)
            resnet_loading.text("✅ ResNet classifier loaded")
            
            trocr_loading = st.empty()
            trocr_loading.text("Loading TrOCR model...")
            self.trocr_ocr = TrOCROCR(device)
            trocr_loading.text("✅ TrOCR model loaded")
            
        st.sidebar.success("All models loaded successfully!")

    def process_image(self, image):
        """Main OCR processing pipeline"""
        orig_image = image.convert("RGB")
        img_with_boxes = orig_image.copy()

        # Detect text regions with YOLO
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Detecting text regions with YOLO...")
        all_boxes = self.yolo_detector.detect_text(orig_image)
        progress_bar.progress(30)
        
        # Flatten and sort boxes if needed
        sorted_boxes = [box for sublist in all_boxes for box in sublist]
        progress_bar.progress(40)

        final_text = []
        classification_results = []
        draw = ImageDraw.Draw(img_with_boxes)
        
        status_text.text("Classifying text regions and performing OCR...")
        total_boxes = len(sorted_boxes)
        
        for i, box in enumerate(sorted_boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))
            classifier_input = preprocess_image(cropped_region)

            class_id, confidence = self.resnet_classifier.classify(classifier_input)
            classification_results.append((x1, y1, x2, y2, class_id, confidence))

            if class_id == 0:  # If not strike-through text
                ocr_text = self.trocr_ocr.recognize_text(cropped_region)
                final_text.append(ocr_text)
                
            # Update progress based on how many boxes we've processed
            progress_value = 40 + (i / total_boxes) * 50
            progress_bar.progress(int(progress_value))

        # Visualization
        for x1, y1, x2, y2, class_id, conf in classification_results:
            color = "green" if class_id == 0 else "red"
            label = f"{'Text' if class_id == 0 else 'Strike'} ({conf:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1+5, y1-25), label, fill=color)

        progress_bar.progress(100)
        status_text.text("Processing complete!")
        
        return img_with_boxes, ' '.join(final_text)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = OCRPipeline()

# File uploader
st.subheader("Upload an image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    # Process button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            # Process the image
            annotated_image, extracted_text = st.session_state.pipeline.process_image(image)
            
            # Display results
            with col2:
                st.subheader("Processed Image")
                st.image(annotated_image, use_column_width=True)
            
            st.subheader("Extracted Text")
            st.write(extracted_text)
            
            # Option to download the annotated image
            buf = io.BytesIO()
            annotated_image.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Annotated Image",
                data=byte_im,
                file_name="annotated_image.jpg",
                mime="image/jpeg",
            )

# Add a section for trying sample images
st.sidebar.markdown("---")
st.sidebar.subheader("Try Sample Images")

sample_images = {
    "Sample 1": "test_images/try1.jpg",
    "Sample 2": "test_images/try2.jpg"
}

selected_sample = st.sidebar.selectbox("Select a sample image", list(sample_images.keys()))

if st.sidebar.button("Process Sample"):
    sample_path = sample_images[selected_sample]
    
    # Check if the sample file exists
    if os.path.exists(sample_path):
        image = Image.open(sample_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sample Image")
            st.image(image, use_column_width=True)
        
        with st.spinner("Processing sample image..."):
            # Process the image
            annotated_image, extracted_text = st.session_state.pipeline.process_image(image)
            
            # Display results
            with col2:
                st.subheader("Processed Image")
                st.image(annotated_image, use_column_width=True)
            
            st.subheader("Extracted Text")
            st.write(extracted_text)
    else:
        st.sidebar.error(f"Sample file {sample_path} not found!")

# Add additional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
This application identifies text in medical forms and distinguishes between normal text and strike-through text.
- Green boxes: Valid text (OCR performed)
- Red boxes: Strike-through text (ignored for OCR)
""")