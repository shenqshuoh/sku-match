#!/usr/bin/env python3
"""Run detection on all test images, save JSON + annotated images.

Usage: python scripts/test_detection.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

API_URL = "http://localhost:8000"
API_KEY = "4ELxhB_knxPmcYfOo4n4VtOmiXAx0WonzgI142VXrIA"
TEST_DIR = Path("/opt/drink_imgs/test")
OUT_DIR = Path("/root/sku-match/test")
ORIGIN_DOMAIN = "http://iovip-z2.qiniuio.com"
CDN_HOST = "vr.jihaihotpot.com"
CDN_PREFIX = "https://vr.jihaihotpot.com/"


def main():
    # Clean old results
    if OUT_DIR.exists():
        for f in OUT_DIR.iterdir():
            f.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in TEST_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in images:
        name = img_path.stem
        task_id = f"{name}_{int(time.time())}"
        print(f"=== {name} ===")

        # Run detection
        payload = json.dumps({"taskId": task_id, "files": str(img_path)})
        raw_json = OUT_DIR / f"{name}.json"

        result = subprocess.run(
            ["curl", "-sf", "-o", str(raw_json), "-w", "%{http_code}",
             "-X", "POST", f"{API_URL}/api/v1/recognition/detect",
             "-H", f"X-API-Key: {API_KEY}",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True,
        )
        http_code = result.stdout.strip()

        if http_code != "200":
            print(f"  FAIL: HTTP {http_code}")
            if raw_json.exists():
                print(f"  {raw_json.read_text()[:200]}")
            continue

        # Parse & pretty-print response
        data = json.loads(raw_json.read_text())
        raw_json.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        if data.get("code") != 1 or not data.get("data"):
            print(f"  FAIL: {data.get('msg', 'unknown error')}")
            continue

        detections = data["data"].get("detections", [])
        matched_image = data["data"].get("matched_image", "")
        print(f"  {len(detections)} detections -> {raw_json}")

        # Download annotated image
        if matched_image and matched_image.startswith(CDN_PREFIX):
            key = matched_image[len(CDN_PREFIX):]
            annotated = OUT_DIR / f"{name}_annotated.jpg"
            dl = subprocess.run(
                ["curl", "-sf", "-o", str(annotated),
                 "-H", f"Host: {CDN_HOST}",
                 f"{ORIGIN_DOMAIN}/{key}"],
                capture_output=True,
            )
            if dl.returncode == 0:
                print(f"  annotated -> {annotated}")
            else:
                print("  WARN: failed to download annotated image")

        print()

    print(f"Done. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
