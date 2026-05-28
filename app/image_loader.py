import base64
import binascii
from io import BytesIO
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError


class ImageLoadError(ValueError):
    pass


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_data_url(value: str) -> bool:
    return value.startswith("data:image/") and ";base64," in value


async def load_image(value: str, timeout: float) -> Image.Image:
    if is_url(value):
        return await _load_url_image(value, timeout)
    if is_data_url(value):
        value = value.split(",", 1)[1]
    return _load_base64_image(value)


async def _load_url_image(url: str, timeout: float) -> Image.Image:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ImageLoadError(f"failed to fetch image URL: {exc}") from exc
    content_type = response.headers.get("content-type", "")
    if content_type and not content_type.startswith("image/"):
        raise ImageLoadError("image URL did not return an image content type")
    return _open_image(response.content)


def _load_base64_image(payload: str) -> Image.Image:
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageLoadError("image must be an HTTP URL, data URL, or valid base64") from exc
    return _open_image(raw)


def _open_image(raw: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageLoadError("image payload is not a valid image") from exc
