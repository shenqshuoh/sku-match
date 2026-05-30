#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.embedder import DEFAULT_EMB_MODEL, Embedder
from src.indexer import SKUIndexer
from src.types import SKUReference
from src.utils import detect_device as _detect_device


def build_reference_index(
    reference_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_EMB_MODEL,
    device: str | None = None,
    batch_size: int = 16,
) -> None:
    if device is None:
        device = _detect_device()
    embedder = Embedder(model_name=model_name, device=device)

    all_image_paths: list[Path] = []
    all_sku_ids: list[str] = []

    for sku_dir in sorted(reference_dir.iterdir()):
        if not sku_dir.is_dir():
            continue

        sku_id = sku_dir.name
        image_paths = (
            list(sku_dir.glob("*.jpg")) + list(sku_dir.glob("*.jpeg")) + list(sku_dir.glob("*.png"))
        )

        for img_path in sorted(image_paths):
            all_image_paths.append(img_path)
            all_sku_ids.append(sku_id)

        print(f"Found {sku_id}: {len(image_paths)} reference images")

    references: list[SKUReference] = []

    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i : i + batch_size]
        batch_sku_ids = all_sku_ids[i : i + batch_size]

        images = [Image.open(p).convert("RGB") for p in batch_paths]
        embeddings = embedder.embed_batch(images)

        for sku_id, img_path, embedding in zip(batch_sku_ids, batch_paths, embeddings):
            references.append(
                SKUReference(
                    sku_id=sku_id,
                    sku_name=sku_id,
                    image_path=img_path,
                    embedding=embedding,
                )
            )

    indexer = SKUIndexer()
    indexer.init_collection(persist_dir=output_dir)
    indexer.build(references)
    indexer.set_emb_model(model_name)

    total_refs = len(references)
    unique_skus = len({r.sku_id for r in references})
    print(f"\nIndex built: {unique_skus} SKUs, {total_refs} total reference images")
    print(f"Model: {model_name}, Dim: {embedder.dim}")
    print(f"Saved to Chroma: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build SKU reference index with DINOv2 embeddings")
    parser.add_argument(
        "-r",
        "--reference-dir",
        type=Path,
        default="data/references/",
        help="Directory containing SKU subdirectories (e.g., data/references/sku_id/*.jpg)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="chroma_data/",
        help="Output directory for the Chroma index",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_EMB_MODEL,
        help=f"Path to local embedding model directory (default: {DEFAULT_EMB_MODEL})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16)",
    )

    args = parser.parse_args()

    if not args.reference_dir.exists():
        print(f"Error: Reference directory not found: {args.reference_dir}", file=sys.stderr)
        sys.exit(1)

    build_reference_index(
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
