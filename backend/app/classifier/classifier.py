from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.classifier.cnn import CNN

CLASSIFICATIONS = {0: "No tumor", 1: "Benign", 2: "Malignant"}

model = CNN()
model.load_state_dict(torch.load("app/classifier/BREAST_CANCER_BEST.pth"))

data_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
        transforms.ToTensor(),  # Convert to tensor
    ]
)


def interpret_prediction(tensor: torch.Tensor) -> tuple[str, float]:
    probs = tensor.squeeze()
    class_idx = torch.argmax(probs).item()
    confidence = probs[class_idx].item()

    return CLASSIFICATIONS[class_idx], confidence


def classify(img_bytes: BytesIO) -> tuple[str, float]:
    """Classify cancer from give image bytes."""
    image = Image.open(img_bytes).convert("RGB")
    image = data_transforms(image)
    image = image.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        outputs = model(image)

    # tensor([[8.6507e-04, 9.8845e-03, 9.8925e-01]])
    confidence_values = F.softmax(outputs.data, dim=1)
    print(confidence_values)
    return interpret_prediction(confidence_values)


# image = "b_c4.png"
# print(classify(model, image))
