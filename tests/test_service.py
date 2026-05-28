import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.qwen_embedding import EmbeddingResult, apply_dimensions, build_messages, get_embedder
from app.schemas import MultiModalInput, normalize_input


class MockEmbedder:
    def __init__(self):
        self.calls: list[tuple[list[MultiModalInput], int | None]] = []

    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> EmbeddingResult:
        self.calls.append((items, dimensions))
        base = [[1.0, 2.0, 3.0] for _ in items]
        return EmbeddingResult(
            embeddings=[apply_dimensions(item, dimensions) for item in base],
            prompt_tokens=len(items),
            total_tokens=len(items),
        )


@pytest.fixture
def mock_embedder():
    embedder = MockEmbedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield embedder
    app.dependency_overrides.clear()


@pytest.fixture
def client(mock_embedder: MockEmbedder) -> TestClient:
    return TestClient(app)


def test_normalize_single_text():
    items = normalize_input("一只白猫")
    assert len(items) == 1
    assert items[0].text == "一只白猫"
    assert items[0].image is None


def test_normalize_batch_text():
    items = normalize_input(["春天", "秋天"])
    assert [item.text for item in items] == ["春天", "秋天"]


def test_build_image_messages():
    messages = build_messages(MultiModalInput(image="https://img.url/cat.jpg"))
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://img.url/cat.jpg"}},
                {"type": "text", "text": "Represent the user's input."},
            ],
        }
    ]


def test_build_mixed_messages():
    messages = build_messages(MultiModalInput(text="一只白猫", image="data:image/jpeg;base64,xxxx"))
    assert messages[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xxxx"}},
        {"type": "text", "text": "一只白猫"},
    ]


def test_apply_dimensions():
    assert apply_dimensions([1.0, 2.0, 3.0], 2) == [1.0, 2.0]
    with pytest.raises(ValueError):
        apply_dimensions([1.0], 2)


def test_embeddings_text_response(client: TestClient, mock_embedder: MockEmbedder):
    response = client.post(
        "/v1/embeddings",
        json={"model": get_settings().model_alias, "input": "一只白色的小猫", "dimensions": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == get_settings().model_alias
    assert body["data"] == [{"object": "embedding", "embedding": [1.0, 2.0], "index": 0}]
    assert body["usage"] == {"prompt_tokens": 1, "total_tokens": 1}
    assert mock_embedder.calls[0][0][0].text == "一只白色的小猫"
    assert mock_embedder.calls[0][1] == 2


def test_embeddings_mixed_batch_response(client: TestClient, mock_embedder: MockEmbedder):
    response = client.post(
        "/v1/embeddings",
        json={
            "model": get_settings().model_alias,
            "input": [
                {"text": "一只白猫", "image": "data:image/jpeg;base64,xxxx"},
                {"image": "https://img.url/mountain.jpg"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["data"]) == 2
    assert [item["index"] for item in body["data"]] == [0, 1]
    assert mock_embedder.calls[0][0][0].text == "一只白猫"
    assert mock_embedder.calls[0][0][1].image == "https://img.url/mountain.jpg"


def test_reject_wrong_model(client: TestClient):
    response = client.post("/v1/embeddings", json={"model": "wrong", "input": "text"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "model"


def test_reject_empty_item(client: TestClient):
    response = client.post("/v1/embeddings", json={"model": get_settings().model_alias, "input": [{}]})
    assert response.status_code == 400
    assert "error" in response.json()
