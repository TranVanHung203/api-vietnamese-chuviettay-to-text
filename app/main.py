import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


SUPPORTED_MODELS = {
    "vgg_transformer",
    "vgg_seq2seq",
    "resnet_transformer",
    "resnet_fpn_transformer",
}

APP_DIR = Path(__file__).resolve().parent
UI_FILE = APP_DIR / "static" / "index.html"


class OCRResponse(BaseModel):
    text: str
    probability: Optional[float] = None
    model: str
    device: str


class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Raw base64 or data URI")
    return_prob: bool = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_device() -> str:
    forced_device = os.getenv("VIETOCR_DEVICE")
    if forced_device:
        return forced_device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def _decode_base64_image(raw_value: str) -> bytes:
    payload = raw_value.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc


class OCRService:
    def __init__(self) -> None:
        model_name = os.getenv("VIETOCR_MODEL", "vgg_transformer")
        if model_name not in SUPPORTED_MODELS:
            supported = ", ".join(sorted(SUPPORTED_MODELS))
            raise RuntimeError(
                f"Unsupported model '{model_name}'. Supported values: {supported}"
            )

        beamsearch = _env_bool("VIETOCR_BEAMSEARCH", True)
        device = _resolve_device()

        config = Cfg.load_config_from_name(model_name)
        config["device"] = device
        config["cnn"]["pretrained"] = False
        config["predictor"]["beamsearch"] = beamsearch

        self.model_name = model_name
        self.device = device
        self.beamsearch = beamsearch
        self.predictor = Predictor(config)

    def recognize(self, image: Image.Image, return_prob: bool) -> OCRResponse:
        if return_prob:
            text, prob = self.predictor.predict(image, return_prob=True)
            probability = None if prob is None else float(prob)
        else:
            text = self.predictor.predict(image, return_prob=False)
            probability = None
        return OCRResponse(
            text=text,
            probability=probability,
            model=self.model_name,
            device=self.device,
        )


@lru_cache(maxsize=1)
def get_ocr_service() -> OCRService:
    return OCRService()


app = FastAPI(
    title="VietOCR Handwriting API",
    version="1.0.0",
    description=(
        "OCR API for Vietnamese handwritten/printed line images using VietOCR."
    ),
)


@app.on_event("startup")
def warmup_model() -> None:
    get_ocr_service()


@app.get("/health")
def health() -> dict:
    service = get_ocr_service()
    return {
        "status": "ok",
        "model": service.model_name,
        "device": service.device,
        "beamsearch": service.beamsearch,
    }


@app.get("/", include_in_schema=False)
def test_ui() -> FileResponse:
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(UI_FILE)


@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_from_file(
    file: UploadFile = File(...),
    return_prob: bool = False,
) -> OCRResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    image = _open_image(content)
    service = get_ocr_service()
    return service.recognize(image=image, return_prob=return_prob)


@app.post("/ocr/base64", response_model=OCRResponse)
def ocr_from_base64(payload: OCRBase64Request) -> OCRResponse:
    image_bytes = _decode_base64_image(payload.image_base64)
    image = _open_image(image_bytes)
    service = get_ocr_service()
    return service.recognize(image=image, return_prob=payload.return_prob)
