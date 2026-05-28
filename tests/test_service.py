import base64
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import get_settings
from app.image_loader import load_image
from app.main import app
from app.qwen_embedding import apply_dimensions, build_prompt, get_embedder
from app.schemas import MultiModalInput, normalize_input


class MockEmbedder:
    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> list[list[float]]:
        base = [[1.0, 2.0, 3.0] for _ in items]
        return [apply_dimensions(item, dimensions) for item in base]


@pytest.fixture(autouse=True)
def override_embedder():
    app.dependency_overrides[get_embedder] = lambda: MockEmbedder()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def image_data_url() -> str:
    image = Image.new("RGB", (1, 1), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def test_normalize_single_text():
    items = normalize_input("一只白猫")
    assert len(items) == 1
    assert items[0].text == "一只白猫"
    assert items[0].image is None


def test_normalize_batch_text():
    items = normalize_input(["春天", "秋天"])
    assert [item.text for item in items] == ["春天", "秋天"]


def test_build_prompts():
    text_prompt = build_prompt(text="一只白猫")
    image_prompt = build_prompt(has_image=True)
    mixed_prompt = build_prompt(text="一只白猫", has_image=True)

    assert "Represent the user's input." in text_prompt
    assert "一只白猫" in text_prompt
    assert "<|vision_start|><|image_pad|><|vision_end|>" in image_prompt
    assert "<|vision_start|><|image_pad|><|vision_end|>一只白猫" in mixed_prompt


@pytest.mark.asyncio
async def test_load_data_url_image():
    image = await load_image(image_data_url(), timeout=1)
    assert image.size == (1, 1)
    assert image.mode == "RGB"


def test_apply_dimensions():
    assert apply_dimensions([1.0, 2.0, 3.0], 2) == [1.0, 2.0]
    with pytest.raises(ValueError):
        apply_dimensions([1.0], 2)


def test_embeddings_text_response(client: TestClient):
    response = client.post(
        "/v1/embeddings",
        json={"model": get_settings().model_alias, "input": "一只白色的小猫", "dimensions": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == get_settings().model_alias
    assert body["data"] == [{"object": "embedding", "embedding": [1.0, 2.0], "index": 0}]
    assert body["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


def test_embeddings_mixed_batch_response(client: TestClient):
    response = client.post(
        "/v1/embeddings",
        json={
            "model": get_settings().model_alias,
            "input": [
                {"text": "一只白猫", "image": image_data_url()},
                {"image": image_data_url()},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert [item["index"] for item in body["data"]] == [0, 1]


def test_reject_wrong_model(client: TestClient):
    response = client.post("/v1/embeddings", json={"model": "wrong", "input": "text"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "model"


def test_reject_empty_item(client: TestClient):
    response = client.post("/v1/embeddings", json={"model": get_settings().model_alias, "input": [{}]})
    assert response.status_code == 400
    assert "error" in response.json()
