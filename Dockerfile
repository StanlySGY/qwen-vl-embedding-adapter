FROM vllm/vllm-openai:latest

ARG MODEL_ID=Qwen/Qwen3-VL-Embedding-2B
ARG HF_ENDPOINT=https://huggingface.co

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface/hub \
    MODEL_ID=${MODEL_ID}

RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir="/models/qwen3-vl-embedding",
    local_dir_use_symlinks=False,
)
PY

COPY app ./app

ENV MODEL_ALIAS=qwen3-vl-embedding \
    MODEL_ID=/models/qwen3-vl-embedding \
    HOST=0.0.0.0 \
    PORT=8080 \
    MAX_MODEL_LEN=8192 \
    SEED=0 \
    IMAGE_FETCH_TIMEOUT=20 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_USE_MODELSCOPE=False

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
