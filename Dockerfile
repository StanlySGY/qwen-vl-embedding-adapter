FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV MODEL_ALIAS=qwen3-vl-embedding \
    UPSTREAM_BASE_URL=http://127.0.0.1:8000 \
    UPSTREAM_MODEL=qwen3-vl-embedding \
    UPSTREAM_TIMEOUT=120 \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
