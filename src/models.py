import torch
import torch.nn as nn
from torchvision import models

class CustomUltrasoundCNN(nn.Module):
    """
    A custom Convolutional Neural Network baseline built from scratch.
    Features 4 Convolutional blocks with BatchNorm, ReLU activation, and MaxPooling,
    followed by Adaptive Average Pooling and Dropout to prevent overfitting.
    """
    def __init__(self, num_classes=2, in_channels=3):
        super(CustomUltrasoundCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name="resnet50", num_classes=2, pretrained=True):
    """
    Factory function to instantiate models.
    Supports transfer learning (ResNet50, EfficientNet-B0) and custom baseline CNN.
    """
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        print(f"Initializing Custom Ultrasound CNN (num_classes={num_classes})")
        model = CustomUltrasoundCNN(num_classes=num_classes)

    elif model_name == "resnet50":
        print(f"Initializing ResNet-50 (pretrained={pretrained}, num_classes={num_classes})")
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        
        # Replace classification head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        print(f"Initializing EfficientNet-B0 (pretrained={pretrained}, num_classes={num_classes})")
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Replace classification head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Choose from 'resnet50', 'efficientnet_b0', or 'custom_cnn'.")

    return model
