"""Lazy-loaded singletons for MegaDetector and MiewID."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import app_config, settings

logger = logging.getLogger(__name__)

_detector = None
_embedder = None
_preprocess = None
_ml_available: bool | None = None


def ml_available() -> bool:
    """Return True if ML dependencies are installed and enabled in config."""
    global _ml_available
    if _ml_available is not None:
        return _ml_available

    if not app_config.models.enabled or not settings.use_ml_models:
        _ml_available = False
        return False

    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401
        from PytorchWildlife.models import detection as pw_detection  # noqa: F401
    except ImportError:
        logger.warning("ML dependencies not installed; falling back to heuristics/mock matching.")
        _ml_available = False
        return False

    _ml_available = True
    return True


def warmup_models() -> None:
    """Load models once at pipeline start."""
    if not ml_available():
        return
    get_detector()
    get_embedder()


def get_detector():
    global _detector
    if _detector is None:
        from PytorchWildlife.models import detection as pw_detection

        device = app_config.models.device
        _detector = pw_detection.MegaDetectorV6(
            device=device,
            version=app_config.models.megadetector_version,
        )
        logger.info(
            "MegaDetector V6 loaded (%s) on %s",
            app_config.models.megadetector_version,
            device,
        )
    return _detector


def get_embedder():
    global _embedder
    if _embedder is None:
        import torch
        from transformers import AutoModel

        model_name = app_config.models.miewid_model
        _embedder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _embedder.eval()
        device = app_config.models.device
        if device != "cpu" and torch.cuda.is_available():
            _embedder = _embedder.to(device)
        logger.info("MiewID loaded: %s", model_name)
    return _embedder


def _get_preprocess():
    global _preprocess
    if _preprocess is None:
        import torchvision.transforms as T

        _preprocess = T.Compose(
            [
                T.Resize((440, 440)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return _preprocess


def run_megadetector(path: Path):
    detector = get_detector()
    return detector.single_image_detection(str(path))


def embed_flank_image(path: Path) -> list[float]:
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    tensor = _get_preprocess()(img).unsqueeze(0)
    device = app_config.models.device
    if device != "cpu" and torch.cuda.is_available():
        tensor = tensor.to(device)

    with torch.no_grad():
        output = get_embedder()(tensor)
        vec = output.detach().cpu().numpy().flatten()

    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec.tolist()
    return (vec / norm).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb))
