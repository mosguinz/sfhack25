import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, classes: int):
        """
        classes: number of classes
        """
        super().__init__()
        # Load pretrained EfficientNetB0
        self.base_model = models.efficientnet_b0(pretrained=True)

        # Replace the first conv layer to accept 1-channel input
        # self.base_model.features[0][0] = nn.Conv2d(
        #     1, 3, kernel_size=1, stride=1, padding=0  # Conv2D(1, 3, (1, 1)) in Keras
        # )

        # Global Average Pooling after features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Custom dense blocks
        self.fc1 = nn.Linear(1280, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(256, classes)

    def forward(self, x):
        x = self.base_model.features(x)  # EfficientNet features
        x = self.pool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)

        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x  # Raw logits (apply softmax during evaluation if needed)
