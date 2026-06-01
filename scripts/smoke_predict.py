#!/usr/bin/env python3
"""Local smoke test for POST /v1/predict."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test food-vision /v1/predict")
    parser.add_argument(
        "--url",
        default=os.environ.get("FOOD_VISION_URL", "http://127.0.0.1:8000"),
        help="Base URL (default: FOOD_VISION_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("FOOD_VISION_API_KEY"),
        help="API key (default: FOOD_VISION_API_KEY env)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pizza.jpg",
        help="Path to JPEG fixture",
    )
    args = parser.parse_args()

    if not args.key:
        print("Set FOOD_VISION_API_KEY or pass --key", file=sys.stderr)
        return 1

    if not args.image.is_file():
        print(f"Image not found: {args.image}", file=sys.stderr)
        print("Run: python scripts/download_fixture.py", file=sys.stderr)
        return 1

    image_b64 = base64.b64encode(args.image.read_bytes()).decode("ascii")
    payload = json.dumps({"imageBase64": image_b64}).encode("utf-8")

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/predict",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Food-Vision-Key": args.key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            print(json.dumps(body, indent=2))
            top = body.get("concepts", [{}])[0]
            if float(top.get("confidence", 0)) > 0.5:
                return 0
            print("Warning: top confidence <= 0.5", file=sys.stderr)
            return 0
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        return exc.code
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc.reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
