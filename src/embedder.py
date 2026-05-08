import ssl
import warnings
import sys
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.utils import detect_device, disable_ssl_verification

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
        use_onnx: bool = False,
    ):
        if device is None:
            device = detect_device()

        self.model_name = model_name
        self.device = device
        self.dim = DIMENSIONS[model_name]
        logger.info("DINOv2Embedder initialized: model=%s, device=%s, dim=%d", model_name, device, self.dim)

        # Prefer ONNX for GPU inference when enabled; fall back to PyTorch
        self._ort_session = None
        self.model = None

        _onnx_path = Path(__file__).parent.parent / "models" / f"{model_name}.onnx"
        if use_onnx and _onnx_path.exists():
            try:
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider"] if device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self._ort_session = ort.InferenceSession(
                    str(_onnx_path),
                    providers=providers,
                )
                logger.info(
                    "DINOv2 ONNX runtime loaded: %s (providers: %s)",
                    _onnx_path,
                    self._ort_session.get_providers(),
                )
            except ImportError:
                logger.info("onnxruntime not installed, falling back to PyTorch")

        if self._ort_session is None:
            # PyTorch fallback — load from vendored dinov2 code + local weights
            _vendor_dir = Path(__file__).parent.parent / "vendor" / "dinov2"
            _weight_path = (
                Path(__file__).parent.parent / "models" / f"{model_name}_pretrain.pth"
            )

            if _weight_path.exists():
                self.model = torch.hub.load(
                    str(_vendor_dir),
                    model_name,
                    source="local",
                    pretrained=False,
                )
                state_dict = torch.load(
                    _weight_path, map_location="cpu", weights_only=True
                )
                self.model.load_state_dict(state_dict, strict=True)
            else:
                # Fallback: download from internet (requires network)
                if sys.platform == "darwin":
                    disable_ssl_verification()
                self.model = torch.hub.load(
                    str(_vendor_dir),
                    model_name,
                    source="local",
                    trust_repo=True,
                )
            self.model.eval()
            self.model.to(device)
            if device == "cuda":
                self.model.half()

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

        if self._ort_session is not None:
            # ONNX Runtime path — faster, less GPU memory
            np_input = tensors.numpy()
            input_name = self._ort_session.get_inputs()[0].name
            outputs = self._ort_session.run(None, {input_name: np_input})
            features = torch.from_numpy(outputs[0])
        else:
            # PyTorch fallback — self.model is always set when _ort_session is None
            assert self.model is not None
            tensors = tensors.to(self.device)
            if self.device == "cuda":
                tensors = tensors.half()
            with torch.no_grad():
                features = self.model(tensors)

        features = features / features.norm(dim=1, keepdim=True)
        return features.cpu().numpy()

    def embed_path(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.embed(image)
