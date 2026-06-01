#!/usr/bin/env python3
"""Download ONNX model weights at Docker build time (reduces runtime RAM spike)."""

from __future__ import annotations

import os
import sys

from huggingface_hub import hf_hub_download

MODEL_ID = os.environ.get("MODEL_ID", "onnx-community/swin-finetuned-food101-ONNX")
ONNX_MODEL_FILE = os.environ.get("ONNX_MODEL_FILE", "onnx/model_int8.onnx")


def main() -> int:
    for filename in ("config.json", ONNX_MODEL_FILE):
        path = hf_hub_download(repo_id=MODEL_ID, filename=filename)
        print(f"Downloaded {MODEL_ID}/{filename} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
