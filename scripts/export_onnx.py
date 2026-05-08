"""Export DINOv2 model to ONNX format for faster inference.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model dinov2_vitb14
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for `from src.*` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.embedder import DINOv2Embedder, DINOv2Variant, DIMENSIONS


def export_onnx(model_name: DINOv2Variant, output_dir: Path = Path("models")) -> Path:
    """Export DINOv2 model to ONNX format."""
    embedder = DINOv2Embedder(model_name=model_name, device="cpu")

    output_path = output_dir / f"{model_name}.onnx"
    dummy_input = torch.randn(1, 3, 518, 518)

    class DINOv2FeatureExtractor(torch.nn.Module):
        """Wrapper that calls forward with only the image tensor (no masks arg)."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model(images)

    wrapper = DINOv2FeatureExtractor(embedder.model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(output_path),
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        opset_version=17,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported {model_name} to {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DINOv2 to ONNX")
    parser.add_argument("--model", default="dinov2_vits14", choices=list(DIMENSIONS.keys()))
    parser.add_argument("--output-dir", default="models", type=Path)
    args = parser.parse_args()
    export_onnx(args.model, args.output_dir)
