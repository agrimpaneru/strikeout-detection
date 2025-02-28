# Strikeout Handwritten Word Detection

## Overview

Handwritten text recognition is a challenging problem, especially when dealing with various handwritten styles and artifacts like strikeouts. This project focuses on detecting handwritten words, classifying them as either normal text or struck-out text, and optionally recognizing the text content. By integrating multiple deep learning models, the pipeline effectively processes images of handwritten documents and identifies strike-through text regions.


## How It Works

The system follows a multi-step pipeline for processing an image and extracting information about handwritten words:
<img src="https://github.com/agrimpaneru/strikeout-detection/blob/main/pipeline.png" width="700" />

1. **Text Detection (YOLOv8)**
   - The first step involves detecting text regions using the YOLOv8 object detection model.
   - This model identifies bounding boxes around words, ensuring that we can isolate individual text regions for further analysis.

2. **Preprocessing (TensorFlow Model)**
   - Each detected text region undergoes preprocessing using a TensorFlow-based image enhancement model.
   - The goal is to normalize the text for better classification and recognition.

3. **Strikeout Classification (ResNet-50)**
   - A fine-tuned ResNet-50 model classifies whether a given text region is a valid word or a struck-out word.
   - The classifier outputs a probability score along with the predicted label ("Text" or "Strike").

4. **OCR Recognition (TrOCR)**
   - If a word is classified as valid (not struck-out), the TrOCR (Transformer-based OCR) model is used to recognize the text.
   - This enables full text extraction from handwritten documents.

5. **Visualization & Output**
   - The system overlays bounding boxes on the original image.
   - Green boxes represent recognized words, while red boxes indicate struck-out words.
   - Optionally, a grid of cropped and preprocessed text regions is displayed for inspection.

## Code Breakdown

### 1. `OCRPipeline.py`
This is the main script that coordinates the entire pipeline.
- Loads and initializes all the models.
- Reads an input image and passes it through the detection, preprocessing, classification, and OCR stages.
- Visualizes results by drawing labeled bounding boxes and displaying classification confidence scores.

### 2. `yolo_detector.py`
- Implements a YOLOv8-based text detector.
- Detects bounding boxes around handwritten words.

### 3. `resnet_classifier.py`
- Implements a ResNet-50-based binary classifier to differentiate between normal and struck-out text.
- Uses a softmax activation to provide classification confidence scores.

### 4. `trocr_ocr.py` 
- Uses a TrOCR model for recognizing handwritten words.
- Only applies OCR if the classifier determines that the text is not struck-out.

### 5. `box_sorting.py`
- A utility function to sort bounding boxes, ensuring that words appear in correct reading order.

### 6. `utils/preprocessing.py`
- Contains the TensorFlow-based image preprocessing pipeline for enhancing text visibility.



## Results
Below are the sample results of the strikeout word detection pipeline:
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/agrimpaneru/strikeout-detection/blob/main/image1.png" width="400" />
    <img src="https://github.com/agrimpaneru/strikeout-detection/blob/main/image2.png" width="400" />
</div>
<img src="https://github.com/agrimpaneru/strikeout-detection/blob/main/image3.png" width="700" />


This is an ongoing project, and I am currently working on refining and improving the pipeline.
