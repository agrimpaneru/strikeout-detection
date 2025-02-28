import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import numpy as np
import tensorflow as tf

from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trocr_ocr import TrOCROCR
from utils.box_sorting import sort_boxes

# Define the accelerator here
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class OCRPipeline:
    def __init__(self):
        self.yolo_detector = YOLODetector()
        self.resnet_classifier = ResNetClassifier(device)
        # self.trocr_ocr = TrOCROCR(device)
        
        # Load the TensorFlow preprocessing model
        self.tf_model_path = 'saved'
        self.tf_model = tf.saved_model.load(self.tf_model_path)

    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocess text image using a TensorFlow model for enhanced preprocessing.
        
        Parameters:
            image (PIL.Image): Input image.
            target_size (tuple): Target size for resizing.
        
        Returns:
            PIL.Image: The processed image.
        """
        from tensorflow.keras.preprocessing import image as tf_image
        
        # Convert to tensor and preprocess
        if isinstance(image, Image.Image):
            # Resize to the model's expected input size
            img_resized = image.resize((224, 448), Image.BILINEAR)  # Use the model's expected input size
            
            # Convert to numpy array and add batch dimension
            img_array = tf_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize to [0, 1]
            
            # Run inference
            output = self.tf_model(img_array)
            
            # Convert output tensor to numpy array
            output_image = output.numpy()
            
            # Extract the first channel (assuming this is the desired output)
            output_channel = output_image[0, :, :, 0]
            
            # Convert to PIL Image (normalize to 0-255 range)
            output_channel = (output_channel * 255).astype(np.uint8)
            processed_image = Image.fromarray(output_channel)
            
            # Resize to the target size while maintaining aspect ratio
            processed_image.thumbnail(target_size, Image.BILINEAR)
            
            # Convert to RGB for compatibility with the rest of the pipeline
            processed_image = processed_image.convert("RGB")
            
            return processed_image
        else:
            raise TypeError("Input must be a PIL Image")

    def process_image(self, image_path, visualize=True):
        """Main OCR processing pipeline"""
        orig_image = Image.open(image_path).convert("RGB")
        img_with_boxes = orig_image.copy()

        # Display input image if visualize is True
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(orig_image)
            plt.title("Input Image", fontsize=16)
            plt.axis('off')
            plt.show()

        # Detect text regions with YOLO
        all_boxes = self.yolo_detector.detect_text(orig_image)
        # Flatten the list of boxes if needed
        sorted_boxes = [box for sublist in all_boxes for box in sublist]

        final_text = []
        classification_results = []
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Create a list to store preprocessed images and original crops for the grid
        preprocessed_images = []
        original_crops = []
        classification_labels = []

        for box in sorted_boxes:
            x1, y1, x2, y2 = map(int, box)
            cropped_region = orig_image.crop((x1, y1, x2, y2))
            
            # Store the original cropped region
            original_crops.append(cropped_region)
            
            # Get preprocessed image for classifier using the new TensorFlow model method
            classifier_input = self.preprocess_image(cropped_region)
            
            # Store the preprocessed image for grid display
            preprocessed_images.append(classifier_input)
            
            # Get classification result
            class_id, confidence = self.resnet_classifier.classify(classifier_input)
            classification_results.append((x1, y1, x2, y2, class_id, confidence))
            
            # Store classification label for grid display
            label = f"{'Text' if class_id == 0 else 'Strike'} ({confidence:.2f})"
            classification_labels.append(label)

            # if class_id == 0:
            #     ocr_text = self.trocr_ocr.recognize_text(cropped_region)
            #     final_text.append(ocr_text)

        # Display the grid of original crops and preprocessed images
        if visualize and preprocessed_images:
            self._display_image_grid(original_crops, preprocessed_images, classification_labels)

        # Visualization of bounding boxes on original image
        if visualize:
            for x1, y1, x2, y2, class_id, conf in classification_results:
                color = "green" if class_id == 0 else "red"
                label = f"{'Text' if class_id == 0 else 'Strike'} ({conf:.2f})"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1+5, y1-25), label, fill=color)

            plt.figure(figsize=(16, 12))
            plt.imshow(img_with_boxes)
            plt.title("OCR Results - Green: Valid Text, Red: Strike-through", fontsize=18, pad=20)
            plt.axis('off')
            plt.show()

        return ' '.join(final_text)
    
    def _display_image_grid(self, original_images, preprocessed_images, labels):
        """Display a grid of original and preprocessed images with their classification labels"""
        # Calculate grid dimensions - each text region gets 2 images (original + preprocessed)
        n = len(original_images)
        cols = min(4, n)  # Max 4 pairs per row (8 images total)
        rows = math.ceil(n / cols)
        
        # Create the figure - each pair takes 2 subplot positions
        fig = plt.figure(figsize=(cols * 5, rows * 3))
        plt.suptitle("Original Crops vs Preprocessed Images with Classifications", fontsize=16)
        
        # Plot each image pair in the grid
        for i in range(n):
            # Calculate position in grid (each sample takes 2 spots)
            row = i // cols
            col = i % cols
            
            # Original crop
            plt.subplot(rows, cols * 2, row * cols * 2 + col * 2 + 1)
            plt.imshow(original_images[i])
            plt.title(f"Original: {labels[i]}", fontsize=10)
            plt.axis('off')
            
            # Preprocessed image
            plt.subplot(rows, cols * 2, row * cols * 2 + col * 2 + 2)
            
            # Handle PIL Image
            if isinstance(preprocessed_images[i], Image.Image):
                plt.imshow(preprocessed_images[i], cmap='gray')
            # Handle PyTorch Tensor
            elif isinstance(preprocessed_images[i], torch.Tensor):
                if len(preprocessed_images[i].shape) == 2 or preprocessed_images[i].shape[0] == 1:
                    # Single channel tensor
                    plt.imshow(preprocessed_images[i].squeeze(), cmap='gray')
                else:
                    # RGB tensor
                    plt_img = preprocessed_images[i].permute(1, 2, 0).cpu().numpy()
                    plt.imshow(plt_img)
            # Handle Numpy Array
            elif isinstance(preprocessed_images[i], np.ndarray):
                if len(preprocessed_images[i].shape) == 2 or preprocessed_images[i].shape[2] == 1:
                    plt.imshow(preprocessed_images[i], cmap='gray')
                else:
                    plt.imshow(preprocessed_images[i])
            
            plt.title(f"Preprocessed: {labels[i]}", fontsize=10)
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Adjust for suptitle
        plt.show()

if __name__ == "__main__":
    pipeline = OCRPipeline()
    extracted_text = pipeline.process_image("test_images/try2.jpg")
    print("Extracted Text:", extracted_text)