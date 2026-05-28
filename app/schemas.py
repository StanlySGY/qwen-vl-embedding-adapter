from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class MultiModalInput(BaseModel):
    text: str | None = None
    image: str | None = None

    @field_validator("text", "image")
    @classmethod
    def reject_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be empty")
        return value

    @model_validator(mode="after")
    def require_content(self) -> "MultiModalInput":
        if self.text is None and self.image is None:
            raise ValueError("input item must include text or image")
        return self


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | MultiModalInput | list[str | MultiModalInput]
    dimensions: int | None = Field(default=None, gt=0)


def normalize_input(value: str | MultiModalInput | list[str | MultiModalInput]) -> list[MultiModalInput]:
    if isinstance(value, str):
        return [MultiModalInput(text=value)]
    if isinstance(value, MultiModalInput):
        return [value]
    if not value:
        raise ValueError("input must not be empty")
    return [MultiModalInput(text=item) if isinstance(item, str) else item for item in value]


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: Usage = Field(default_factory=Usage)


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def openai_error(message: str, error_type: str = "invalid_request_error", param: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": None}}
