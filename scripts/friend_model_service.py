"""
Friend's tiger comparison model service stub.

Your friend replaces the `compare_tigers` function with their actual model.
Run: uvicorn scripts.friend_model_service:app --port 8001
Then set FRIEND_MODEL_MOCK=false in .env
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Tiger Comparison Model", version="1.0.0")


class CompareResponse(BaseModel):
    same_tiger: bool
    confidence: float
    model_version: str


def compare_tigers(image_a_bytes: bytes, image_b_bytes: bytes) -> tuple[bool, float]:
    """
    REPLACE THIS with your friend's stripe-pattern comparison model.

    Expected return: (same_tiger: bool, confidence: 0.0-1.0)
    """
    import hashlib

    ha = hashlib.md5(image_a_bytes).hexdigest()
    hb = hashlib.md5(image_b_bytes).hexdigest()

    if ha == hb:
        return True, 0.99

    similarity = sum(a == b for a, b in zip(ha, hb)) / len(ha)
    return similarity > 0.6, similarity


@app.post("/compare", response_model=CompareResponse)
async def compare(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
):
    a_bytes = await image_a.read()
    b_bytes = await image_b.read()
    same, confidence = compare_tigers(a_bytes, b_bytes)
    return CompareResponse(
        same_tiger=same,
        confidence=confidence,
        model_version="friend-model-v1",
    )


@app.post("/compare/paths")
async def compare_paths(image_a_path: str, image_b_path: str):
    """Alternative endpoint accepting file paths (for local testing)."""
    a_bytes = Path(image_a_path).read_bytes()
    b_bytes = Path(image_b_path).read_bytes()
    same, confidence = compare_tigers(a_bytes, b_bytes)
    return CompareResponse(
        same_tiger=same,
        confidence=confidence,
        model_version="friend-model-v1",
    )


@app.get("/health")
def health():
    return {"status": "ok"}
