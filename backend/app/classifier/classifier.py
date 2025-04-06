from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.classifier.cnn import CNN

BREAST_CLASSES = {0: "No tumor", 1: "Benign", 2: "Malignant"}
CHEST_CLASSES = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "Mass",
    4: "No Finding",
    5: "Nodule",
    6: "Pneumonia",
}

breast_model = CNN(classes=3)
breast_model.load_state_dict(
    torch.load(
        "app/classifier/BREAST_CANCER_BEST.pth", map_location=torch.device("cpu")
    )
)
chest_model = CNN(classes=7)
chest_model.load_state_dict(
    torch.load("app/classifier/CHEST_XRAY_BEST.pth", map_location=torch.device("cpu"))
)

data_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
        transforms.ToTensor(),  # Convert to tensor
    ]
)


def interpret_prediction(tensor: torch.Tensor, model_type: str) -> tuple[str, float]:
    probs = tensor.squeeze()
    class_idx = torch.argmax(probs).item()
    confidence = probs[class_idx].item()

    classification: dict[int, str]
    if model_type == "breast":
        classification = BREAST_CLASSES
    elif model_type == "chest":
        classification = CHEST_CLASSES
    else:
        raise ValueError("Invalid model type")

    return classification[class_idx], confidence


def classify(img_bytes: BytesIO, model_type: str) -> tuple[str, float]:
    """Classify cancer from give image bytes."""
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    image = data_transforms(image)
    image = image.unsqueeze(0)

    model: CNN
    if model_type == "breast":
        model = breast_model
    elif model_type == "chest":
        model = chest_model
    else:
        raise ValueError("Invalid model type")

    model.eval()

    with torch.no_grad():
        outputs = model(image)

    # tensor([[8.6507e-04, 9.8845e-03, 9.8925e-01]])
    confidence_values = F.softmax(outputs.data, dim=1)
    print(confidence_values)
    return interpret_prediction(confidence_values, model_type)


# image = "b_c4.png"
# print(classify(model, image))
