import os
import torch
import torch.nn as nn
from torchvision import transforms, models
# import kagglehub
from PIL import Image

class ResNetClassifier:
    def __init__(self, device):
        """Load ResNet50 classifier"""
        # path = kagglehub.model_download("/models/ModelsHere/Resnet50.pt")
        
        # if os.path.isdir(path):
        #     files = os.listdir(path)
        #     model_file = next((f for f in files if f.endswith('.pth')), None)
        #     if not model_file:
        #         raise FileNotFoundError(f"No .pth file found in {path}")
        #     path = os.path.join(path, model_file)

        model_path = 'Resnet50.pth'

        self.device = device
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, image):
        """Classify if text region contains strike-through"""
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(prob, dim=1)
        return predicted.item(), confidence.item()
