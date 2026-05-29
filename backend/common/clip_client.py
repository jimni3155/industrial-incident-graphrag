from __future__ import annotations
from email.mime import image
from platform import processor

import torch
from transformers import CLIPModel, CLIPProcessor
import threading

_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_model:     CLIPModel     | None = None
_processor: CLIPProcessor | None = None
_lock = threading.Lock()


def _load() -> tuple[CLIPModel, CLIPProcessor]:
    global _model, _processor

    if _model is None or _processor is None:
        with _lock:
            if _model is None or _processor is None:
                _processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
                _model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
                _model.eval()

    return _model, _processor


def get_clip_image_embedding(image_path: str) -> list[float]:
    from PIL import Image as PILImage

    model, processor = _load()
    with PILImage.open(image_path) as img:
        image = img.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs  = model.get_image_features(**inputs)
        # Depending on the version, returns either a tensor or BaseModelOutputWithPooling
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()


def get_clip_text_embedding(text: str) -> list[float]:
    model, processor = _load()
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        outputs  = model.get_text_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].tolist()