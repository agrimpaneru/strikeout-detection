import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCROCR:
    def __init__(self, device):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten").to(device)

    def recognize_text(self, image):
        """Perform OCR on text region using TrOCR"""
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.model.device)
        generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
