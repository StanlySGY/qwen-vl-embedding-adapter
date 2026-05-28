# Qwen3-VL Embedding Adapter

OpenAI-compatible `/v1/embeddings` adapter for `Qwen/Qwen3-VL-Embedding-2B` with text, image URL, base64 image, mixed input, batch, and dimensions support.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Build online, run offline

The Docker image downloads `Qwen/Qwen3-VL-Embedding-2B` during `docker build` and stores it at `/models/qwen3-vl-embedding`. Runtime sets HuggingFace/Transformers offline flags, so the exported image can be loaded in a fully offline environment without downloading Python dependencies or model files.

```bash
docker build --platform linux/arm64 -t qwen-vl-embedding:offline .
docker save qwen-vl-embedding:offline | gzip > qwen-vl-embedding-offline-arm64.tar.gz
```

Offline target:

```bash
gunzip -c qwen-vl-embedding-offline-arm64.tar.gz | docker load
docker run --gpus all -p 8080:8080 qwen-vl-embedding:offline
```

If your ARM server cannot run `vllm/vllm-openai:latest`, replace the base image with a vLLM image that supports your accelerator/runtime before building.

## API

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-embedding","input":"一只白色的小猫"}'
```
