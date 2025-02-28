import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

class YOLODetector:
    def __init__(self, model_path='yolo.pt'):
        self.model = YOLO(model_path)

    def detect_text(self, image):
        """Detect text regions in an image"""
        results = self.model.predict(image)
        boxes = [box.xyxy.cpu().numpy() for result in results for box in result.boxes]
        return boxes

    def draw_bounding_boxes(self, image_path, output_path="output.jpg"):
        """Detect text and draw bounding boxes"""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        boxes = self.detect_text(image)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[0])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        image.save(output_path)
        image.show()  # Show the image with bounding boxes

#! Uncomment following code to test this individual class.
if __name__ == "__main__":
    detector = YOLODetector()
    detector.draw_bounding_boxes("../test_images/try2.jpg", "output.jpg")
