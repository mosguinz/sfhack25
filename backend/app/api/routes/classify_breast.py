import io
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from app.classifier.classifier import classify
from app.models import ClassificationResult

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("/", response_model=ClassificationResult)
async def classify_image(file: UploadFile = File(...)) -> Any:
    """
    Upload an image to classify breast cancer type.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        diagnosis, confidence = classify(image)
        result = ClassificationResult(diagnosis=diagnosis, confidence=confidence)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Image processing failed: {str(e)}"
        )
