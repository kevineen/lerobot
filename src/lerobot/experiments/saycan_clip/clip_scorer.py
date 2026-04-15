from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class ClipScore:
    text: str
    score: float


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Convert numpy image to a PIL RGB image.

    Accepts:
    - HWC uint8
    - CHW float/uint8
    """
    arr = image
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if float(arr.max()) <= 1.0 else arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class ClipScorer:
    def __init__(
        self,
        pretrained_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = CLIPModel.from_pretrained(pretrained_name)
        self.processor = CLIPProcessor.from_pretrained(pretrained_name, use_fast=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, image: np.ndarray, texts: list[str]) -> list[ClipScore]:
        if not texts:
            return []

        pil = _to_pil_rgb(image)

        # Image embedding
        img_inputs = self.processor(images=[pil], return_tensors="pt")
        img_inputs = {k: v.to(self.device) for k, v in img_inputs.items()}
        image_features = self.model.get_image_features(**img_inputs)

        # Text embeddings (batched)
        all_text_features = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tok = self.processor.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
            tok = {k: v.to(self.device) for k, v in tok.items()}
            feats = self.model.get_text_features(**tok)
            all_text_features.append(feats)
        text_features = torch.cat(all_text_features, dim=0)

        if self.normalize_embeddings:
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        # Similarity: (N,)
        sims = (text_features @ image_features.squeeze(0).unsqueeze(-1)).squeeze(-1)
        sims = sims.detach().float().cpu().numpy()

        scored = [ClipScore(text=t, score=float(s)) for t, s in zip(texts, sims, strict=True)]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

