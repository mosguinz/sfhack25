from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile

from app.classifier.classifier import classify
from app.models import ClassificationResult

router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("/breast", response_model=ClassificationResult)
async def classify_breast_image(file: UploadFile) -> Any:
    """
    Upload an image to classify breast cancer type.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        diagnosis, confidence = classify(image_bytes, "breast")
        return ClassificationResult(diagnosis=diagnosis, confidence=confidence)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Breast image processing failed: {str(e)}"
        )


@router.post("/chest", response_model=ClassificationResult)
async def classify_chest_image(file: UploadFile) -> Any:
    """
    Upload an image to classify chest condition.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        diagnosis, confidence = classify(image_bytes, "chest")
        return ClassificationResult(diagnosis=diagnosis, confidence=confidence)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Chest image processing failed: {str(e)}"
        )


@router.post("/brain", response_model=ClassificationResult)
async def classify_brain_image(file: UploadFile) -> Any:
    """
    Upload an image to classify breast cancer type.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        diagnosis, confidence = classify(image_bytes, "brain")
        return ClassificationResult(diagnosis=diagnosis, confidence=confidence)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Brain image processing failed: {str(e)}"
        )
