FROM python:3.11-slim

# Quantized ONNX uses less RAM on Render Starter (512 MB)
ARG MODEL_ID=onnx-community/swin-finetuned-food101-ONNX
ARG ONNX_MODEL_FILE=onnx/model_int8.onnx

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    MODEL_ID=${MODEL_ID} \
    ONNX_MODEL_FILE=${ONNX_MODEL_FILE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/download_onnx_model.py ./scripts/download_onnx_model.py

# Bake weights into the image so startup does not download + load at once
RUN mkdir -p /app/.cache/huggingface \
    && python scripts/download_onnx_model.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
