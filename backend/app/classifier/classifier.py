from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.classifier.cnn import CNN
from app.classifier.dense_cnn import DenseCNN
from app.classifier.preprocessor import Preprocessor

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
BRAIN_CLASSES = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Finding',
    3: 'Pituitary',
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
cnn_data_transform_pipeline = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize to 299x299 pixels
        transforms.ToTensor(),  # Convert to tensor
    ]
)

brain_model = DenseCNN()
dense_cnn_transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        Preprocessor(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225] # ImageNet std
        )
    ]);

def interpret_prediction(tensor: torch.Tensor, model_type: str) -> tuple[str, float]:
    probs = tensor.squeeze()
    class_idx = torch.argmax(probs).item()
    confidence = probs[class_idx].item()

    classification: dict[int, str]
    if model_type == "breast":
        classification = BREAST_CLASSES
    elif model_type == "chest":
        classification = CHEST_CLASSES
    elif model_type == "brain":
        classification = BRAIN_CLASSES
    else:
        raise ValueError("Invalid model type")

    return classification[class_idx], confidence


def classify(img_bytes: BytesIO, model_type: str) -> tuple[str, float]:
    """Classify cancer from give image bytes."""
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    model = None
    if model_type == "breast":
        model = breast_model
        image = cnn_data_transform_pipeline(image)
    elif model_type == "chest":
        model = chest_model
        image = cnn_data_transform_pipeline(image)
    elif model_type == "brain":
        model = brain_model
        image = dense_cnn_transform_pipeline(image)
    else:
        raise ValueError("Invalid model type")

    image = image.unsqueeze(0)

    model.eval()

    with torch.no_grad():
        outputs = model(image)

    # tensor([[8.6507e-04, 9.8845e-03, 9.8925e-01]])
    confidence_values = F.softmax(outputs.data, dim=1)
    print(confidence_values)
    return interpret_prediction(confidence_values, model_type)


# image = "b_c4.png"
# print(classify(model, image))
