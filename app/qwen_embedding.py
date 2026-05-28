from functools import lru_cache
from typing import Protocol

from PIL import Image

from app.config import Settings, get_settings
from app.image_loader import load_image
from app.schemas import MultiModalInput

DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class Embedder(Protocol):
    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> list[list[float]]:
        ...


def build_prompt(text: str | None = None, has_image: bool = False) -> str:
    user_content = ""
    if has_image:
        user_content += IMAGE_PLACEHOLDER
    if text:
        user_content += text
    return (
        f"<|im_start|>system\n{DEFAULT_INSTRUCTION}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def apply_dimensions(embedding: list[float], dimensions: int | None) -> list[float]:
    if dimensions is None:
        return embedding
    if dimensions > len(embedding):
        raise ValueError(f"dimensions must be <= embedding size {len(embedding)}")
    return embedding[:dimensions]


class QwenVLEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        self._smart_resize = _load_smart_resize()

    async def embed(self, items: list[MultiModalInput], dimensions: int | None = None) -> list[list[float]]:
        inputs = []
        for item in items:
            has_image = item.image is not None
            prompt = build_prompt(text=item.text, has_image=has_image)
            if has_image:
                image = await load_image(item.image or "", self.settings.image_fetch_timeout)
                inputs.append({"prompt": prompt, "multi_modal_data": {"image": self._prepare_image(image)}})
            else:
                inputs.append(prompt)

        outputs = self.llm.embed(inputs, use_tqdm=False)
        embeddings = [list(output.outputs.embedding) for output in outputs]
        return [apply_dimensions(embedding, dimensions) for embedding in embeddings]

    @property
    def llm(self):
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def _create_llm(self):
        from vllm import LLM

        kwargs = {
            "model": self.settings.model_id,
            "runner": "pooling",
            "max_model_len": self.settings.max_model_len,
            "limit_mm_per_prompt": {"image": 1},
            "seed": self.settings.seed,
        }
        if self._smart_resize is not None:
            kwargs["mm_processor_kwargs"] = {"do_resize": False}
        return LLM(**kwargs)

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        if self._smart_resize is None:
            return image
        width, height = image.size
        resized_height, resized_width = self._smart_resize(height, width, factor=32)
        return image.resize((resized_width, resized_height))


def _load_smart_resize():
    try:
        from qwen_vl_utils import smart_resize
    except ModuleNotFoundError:
        return None
    return smart_resize


@lru_cache
def get_embedder() -> QwenVLEmbedder:
    return QwenVLEmbedder(get_settings())
