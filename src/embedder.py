import ssl
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

DINOv2Variant = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]
DIMENSIONS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}

__all__ = ["DINOv2Embedder", "DINOv2Variant", "DIMENSIONS"]

warnings.filterwarnings("ignore", message="xFormers is not available")


class DINOv2Embedder:
    def __init__(
        self,
        model_name: DINOv2Variant = "dinov2_vitb14",
        device: str | None = None,
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model_name = model_name
        self.device = device
        self.dim = DIMENSIONS[model_name]

        ssl._create_default_https_context = ssl._create_unverified_context
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
        )
        self.model.eval()
        self.model.to(device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(518),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def embed(self, image: Image.Image) -> np.ndarray:
        return self.embed_batch([image])

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        tensors = torch.stack([self.transform(img) for img in images])
        tensors = tensors.to(self.device)

        with torch.no_grad():
            features = self.model(tensors)
            features = features / features.norm(dim=1, keepdim=True)

        return features.cpu().numpy()

    def embed_path(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.embed(image)
