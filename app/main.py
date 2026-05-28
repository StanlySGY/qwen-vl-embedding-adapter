from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.qwen_embedding import Embedder, UpstreamEmbeddingError, get_embedder
from app.schemas import EmbeddingData, EmbeddingsRequest, EmbeddingsResponse, Usage, normalize_input, openai_error

app = FastAPI(title="Qwen3-VL Embedding Adapter", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=openai_error(str(exc), param="input"),
    )


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "model": settings.model_alias}


@app.post("/v1/embeddings", response_model=EmbeddingsResponse)
async def create_embeddings(
    request: EmbeddingsRequest,
    settings: Settings = Depends(get_settings),
    embedder: Embedder = Depends(get_embedder),
) -> EmbeddingsResponse | JSONResponse:
    if request.model != settings.model_alias:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=openai_error(f"model must be '{settings.model_alias}'", param="model"),
        )

    try:
        items = normalize_input(request.input)
        result = await embedder.embed(items, request.dimensions)
    except ValueError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=openai_error(str(exc), param="input"),
        )
    except UpstreamEmbeddingError as exc:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=openai_error(str(exc), error_type="upstream_error"),
        )

    return EmbeddingsResponse(
        data=[EmbeddingData(embedding=embedding, index=index) for index, embedding in enumerate(result.embeddings)],
        model=settings.model_alias,
        usage=Usage(prompt_tokens=result.prompt_tokens, total_tokens=result.total_tokens),
    )
