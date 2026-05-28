# Qwen3-VL Embedding Adapter

Lightweight OpenAI-compatible `/v1/embeddings` proxy for an already deployed Qwen3-VL-Embedding service.

The adapter accepts:
- single text
- text batch
- image URL
- image base64 data URL
- mixed text + image batch

Text-only requests are forwarded to the upstream `/v1/embeddings` endpoint unchanged. Image and mixed requests are converted to vLLM chat-embeddings `messages` format.

## Environment

```bash
MODEL_ALIAS=qwen3-vl-embedding
UPSTREAM_BASE_URL=http://your-existing-qwen-service:8000
UPSTREAM_MODEL=qwen3-vl-embedding
UPSTREAM_TIMEOUT=120
```

## Run locally

```bash
pip install -r requirements.txt
UPSTREAM_BASE_URL=http://127.0.0.1:8000 uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Build ARM64 image

```bash
docker build --platform linux/arm64 -t qwen-vl-embedding-adapter:v1.0.0 .
docker save qwen-vl-embedding-adapter:v1.0.0 | gzip > qwen-vl-embedding-adapter-v1.0.0-arm64.tar.gz
```

## Run

```bash
docker run -p 8080:8080 \
  -e UPSTREAM_BASE_URL=http://your-existing-qwen-service:8000 \
  -e UPSTREAM_MODEL=qwen3-vl-embedding \
  qwen-vl-embedding-adapter:v1.0.0
```

## API

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-embedding","input":"一只白色的小猫"}'
```

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-embedding","input":[{"image":"https://img.url/cat.jpg"}]}'
```

```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-vl-embedding","input":[{"text":"一只白猫","image":"data:image/jpeg;base64,xxxx"}]}'
```
