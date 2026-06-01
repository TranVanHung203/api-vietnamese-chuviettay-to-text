import base64
import io
import os
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import cv2
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
MODEL_CACHE_DIR = APP_DIR / "models"
NUMBER_WORDS = {
    "không",
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
    "tư",
    "lăm",
    "mốt",
    "linh",
    "lẻ",
    "mười",
    "mươi",
    "trăm",
    "nghìn",
    "ngàn",
    "triệu",
    "tỷ",
    "chục",
    "đơn vị",
    "nhăm",
}
NUMBER_WORDS_SINGLE = sorted(
    [word for word in NUMBER_WORDS if " " not in word], key=len, reverse=True
)
DON_VI_PLAIN = "don vi"
DONVI_PLAIN = "donvi"
WORD_TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
TOKEN_RE = re.compile(r"\s+|\d+|[^\W\d_]+|[^\s\w]", flags=re.UNICODE)
DIGIT_WORD_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+", flags=re.UNICODE)
DIGIT_PREPROCESS_TARGET_HEIGHT = 64
DIGIT_PREPROCESS_MAX_WIDTH = 1600
DIGIT_PREPROCESS_PADDING = 16
DIGIT_ALLOWED_SYMBOLS = set("+-=:*/.,()")


class OCRResponse(BaseModel):
    text: str
    probability: Optional[float] = None
    lines: Optional[list[str]] = None
    line_probabilities: Optional[list[Optional[float]]] = None
    raw_text: Optional[str] = None
    raw_lines: Optional[list[str]] = None
    number_word_mode: bool = False
    number_digit_mode: bool = False
    model: str
    device: str


class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Raw base64 or data URI")
    return_prob: bool = False
    multiline: bool = True
    number_word_mode: bool = False
    number_digit_mode: bool = False


class DigitVariantItem(BaseModel):
    index: int
    width: int
    height: int
    image_base64: str


class DigitVariantResponse(BaseModel):
    variants: list[DigitVariantItem]


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


def _resize_keep_ratio(
    image_array: np.ndarray, target_height: int, max_width: int
) -> np.ndarray:
    height, width = image_array.shape[:2]
    if height <= 0 or width <= 0:
        return image_array

    scale = target_height / height
    if scale <= 1.0 and width <= max_width:
        return image_array

    new_width = min(max_width, max(1, int(round(width * scale))))
    return cv2.resize(
        image_array,
        (new_width, target_height),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
    )


def _crop_to_ink(binary_image: np.ndarray) -> np.ndarray:
    ink_mask = binary_image < 250
    if int(ink_mask.sum()) < 8:
        return binary_image

    rows = np.where(ink_mask.any(axis=1))[0]
    cols = np.where(ink_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return binary_image

    top = max(0, int(rows[0]) - DIGIT_PREPROCESS_PADDING)
    bottom = min(binary_image.shape[0], int(rows[-1]) + DIGIT_PREPROCESS_PADDING + 1)
    left = max(0, int(cols[0]) - DIGIT_PREPROCESS_PADDING)
    right = min(binary_image.shape[1], int(cols[-1]) + DIGIT_PREPROCESS_PADDING + 1)
    return binary_image[top:bottom, left:right]


def _ensure_light_background(gray: np.ndarray) -> np.ndarray:
    border_pixels = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )
    if float(np.median(border_pixels)) < 127.0:
        return 255 - gray
    return gray


def _normalize_digit_background(gray: np.ndarray) -> np.ndarray:
    min_side = min(gray.shape[:2])
    if min_side < 3:
        return gray

    kernel_size = min(51, max(3, (min_side // 2) | 1))
    background = cv2.medianBlur(gray, kernel_size)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def _deskew_digit_image(binary_image: np.ndarray) -> np.ndarray:
    ink_mask = binary_image < 250
    if int(ink_mask.sum()) < 20:
        return binary_image

    points = np.column_stack(np.where(ink_mask))[:, ::-1].astype(np.float32)
    angle = cv2.minAreaRect(points)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 12:
        return binary_image

    height, width = binary_image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _remove_small_ink_components(binary_image: np.ndarray) -> np.ndarray:
    inverted = 255 - binary_image
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    if component_count <= 1:
        return binary_image

    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = max(6, int(image_area * 0.00015))
    cleaned = np.full(binary_image.shape, 255, dtype=np.uint8)
    for component_id in range(1, component_count):
        x, y, width, height, area = stats[component_id]
        if area < min_area or width < 2 or height < 3:
            continue
        cleaned[labels == component_id] = 0

    if int((cleaned < 250).sum()) < 8:
        return binary_image
    return cleaned


def _finish_digit_binary(binary_image: np.ndarray) -> Image.Image:
    binary_image = _deskew_digit_image(binary_image)
    binary_image = _remove_small_ink_components(binary_image)
    binary_image = _crop_to_ink(binary_image)
    binary_image = cv2.copyMakeBorder(
        binary_image,
        DIGIT_PREPROCESS_PADDING,
        DIGIT_PREPROCESS_PADDING,
        DIGIT_PREPROCESS_PADDING,
        DIGIT_PREPROCESS_PADDING,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    binary_image = _resize_keep_ratio(
        binary_image,
        target_height=DIGIT_PREPROCESS_TARGET_HEIGHT,
        max_width=DIGIT_PREPROCESS_MAX_WIDTH,
    )
    return Image.fromarray(binary_image).convert("RGB")


def _dedupe_digit_variants(images: list[Image.Image]) -> list[Image.Image]:
    variants: list[Image.Image] = []
    seen: set[tuple[int, int, int]] = set()
    for image in images:
        gray = np.array(image.convert("L"))
        key = (image.width, image.height, hash(gray.tobytes()))
        if key in seen:
            continue
        seen.add(key)
        variants.append(image)
    return variants


def _preprocess_digit_variants(image: Image.Image) -> list[Image.Image]:
    gray = np.array(image.convert("L"))
    if gray.size == 0:
        return [image.convert("RGB")]

    gray = _ensure_light_background(gray)
    gray = _resize_keep_ratio(
        gray,
        target_height=DIGIT_PREPROCESS_TARGET_HEIGHT,
        max_width=DIGIT_PREPROCESS_MAX_WIDTH,
    )
    gray = cv2.fastNlMeansDenoising(
        gray, None, h=12, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    normalized = _normalize_digit_background(gray)
    blurred_enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    blurred_normalized = cv2.GaussianBlur(normalized, (3, 3), 0)

    _, otsu = cv2.threshold(
        blurred_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    min_side = min(blurred_normalized.shape[:2])
    adaptive_block_size = min(31, max(3, (min_side // 2) | 1))
    adaptive = cv2.adaptiveThreshold(
        blurred_normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_block_size,
        11,
    )
    _, normalized_otsu = cv2.threshold(
        blurred_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((2, 2), dtype=np.uint8)
    variants = []
    for binary in (otsu, adaptive, normalized_otsu):
        binary = _ensure_light_background(binary)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        variants.extend(
            [
                _finish_digit_binary(closed),
                _finish_digit_binary(opened),
                _finish_digit_binary(dilated),
            ]
        )

    return _dedupe_digit_variants(variants)


def _preprocess_digit_image(image: Image.Image) -> Image.Image:
    return _preprocess_digit_variants(image)[0]


def _decode_base64_image(raw_value: str) -> bytes:
    payload = raw_value.strip()
    if "," in payload and payload.lower().startswith("data:"):
        payload = payload.split(",", 1)[1]
    try:
        return base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Invalid base64 payload") from exc


def _image_to_png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _strip_diacritics(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_marks = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


NUMBER_WORDS_PLAIN = {
    word: _strip_diacritics(word.lower()) for word in NUMBER_WORDS_SINGLE
}
PLAIN_TO_NUMBER_WORDS: dict[str, list[str]] = {}
for word, plain in NUMBER_WORDS_PLAIN.items():
    PLAIN_TO_NUMBER_WORDS.setdefault(plain, []).append(word)

# Ưu tiên dạng chuẩn khi token không dấu bị trùng nhiều từ (vd: mot -> một/mốt).
PLAIN_WORD_PREFERENCE = {
    "mot": "một",
    "muoi": "mười",
    "nam": "năm",
}


def _normalize_number_token(token: str) -> str:
    lowered = token.lower()
    if lowered in NUMBER_WORDS:
        return lowered

    token_plain = _strip_diacritics(lowered)
    if token_plain in PLAIN_TO_NUMBER_WORDS:
        candidates = PLAIN_TO_NUMBER_WORDS[token_plain]
        preferred = PLAIN_WORD_PREFERENCE.get(token_plain)
        if preferred and preferred in candidates:
            return preferred
        return sorted(candidates)[0]

    best_word = lowered
    best_score = 0.0
    for candidate, candidate_plain in NUMBER_WORDS_PLAIN.items():
        score = SequenceMatcher(None, token_plain, candidate_plain).ratio()
        if score > best_score:
            best_score = score
            best_word = candidate

    # Fuzzy threshold vừa đủ để sửa sai OCR phổ biến, tránh sửa quá tay.
    if best_score >= 0.72:
        return best_word
    return lowered


def _looks_like_don_vi_pair(first: str, second: str) -> bool:
    pair_plain = f"{_strip_diacritics(first.lower())} {_strip_diacritics(second.lower())}"
    if pair_plain == DON_VI_PLAIN:
        return True
    return SequenceMatcher(None, pair_plain, DON_VI_PLAIN).ratio() >= 0.78


def _looks_like_don_vi_single(token: str) -> bool:
    token_plain = _strip_diacritics(token.lower())
    if token_plain == DONVI_PLAIN:
        return True
    return SequenceMatcher(None, token_plain, DONVI_PLAIN).ratio() >= 0.82


def _normalize_number_words_line(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return text.strip().lower()

    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.isspace():
            out.append(token)
            i += 1
            continue

        if not WORD_TOKEN_RE.fullmatch(token):
            out.append(token)
            i += 1
            continue

        next_word = None
        j = i + 1
        while j < len(tokens):
            if tokens[j].isspace():
                j += 1
                continue
            next_word = tokens[j]
            break

        if next_word and WORD_TOKEN_RE.fullmatch(next_word) and _looks_like_don_vi_pair(token, next_word):
            out.append("đơn vị")
            i = j + 1
            continue

        if _looks_like_don_vi_single(token):
            out.append("đơn vị")
            i += 1
            continue

        out.append(_normalize_number_token(token))
        i += 1

    return "".join(out).strip()


# Map common Vietnamese number words (with/without marks after strip) to 0-9.
DIGIT_PLAIN_TO_CHAR = {
    "khong": "0",
    "mot": "1",
    "hai": "2",
    "ba": "3",
    "bon": "4",
    "tu": "4",
    "nam": "5",
    "lam": "5",
    "sau": "6",
    "bay": "7",
    "tam": "8",
    "chin": "9",
}
DIGIT_OCR_CONFUSIONS = {
    "o": "0",
    "i": "1",
    "l": "1",
    "|": "1",
    "!": "1",
    "z": "2",
    "s": "5",
    "b": "6",
    "g": "9",
    "q": "9",
}


def _normalize_digit_token(token: str) -> str:
    lowered = token.lower()
    if lowered.isdigit():
        return lowered
    if lowered in DIGIT_OCR_CONFUSIONS:
        return DIGIT_OCR_CONFUSIONS[lowered]

    token_plain = _strip_diacritics(lowered)
    if token_plain in DIGIT_PLAIN_TO_CHAR:
        return DIGIT_PLAIN_TO_CHAR[token_plain]

    # In digit mode, always force word-like tokens to the closest 0-9 candidate.
    best_digit = "0"
    best_score = -1.0
    for candidate_plain, digit_char in DIGIT_PLAIN_TO_CHAR.items():
        score = SequenceMatcher(None, token_plain, candidate_plain).ratio()
        if score > best_score:
            best_score = score
            best_digit = digit_char

    return best_digit


def _normalize_number_digits_line(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return text.strip().lower()

    out: list[str] = []
    for token in tokens:
        if token.isspace():
            out.append(token)
            continue

        if token.isdigit() or WORD_TOKEN_RE.fullmatch(token):
            out.append(_normalize_digit_token(token))
            continue

        out.append(DIGIT_OCR_CONFUSIONS.get(token.lower(), token))

    return "".join(out).strip()


def _score_digit_prediction(text: str, probability: Optional[float]) -> float:
    normalized = _normalize_number_digits_line(text)
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return -1.0

    digit_count = sum(ch.isdigit() for ch in compact)
    allowed_count = sum(
        ch.isdigit() or ch in DIGIT_ALLOWED_SYMBOLS for ch in compact
    )
    bad_count = len(compact) - allowed_count
    score = 0.0 if probability is None else probability
    score += min(digit_count, 24) * 0.03
    score -= bad_count * 0.12
    if digit_count == 0:
        score -= 0.5
    return score


def _normalize_text_by_number_mode(
    text: str, number_word_mode: bool, number_digit_mode: bool
) -> tuple[str, Optional[str]]:
    if number_digit_mode:
        normalized_text = _normalize_number_digits_line(text)
    elif number_word_mode:
        normalized_text = _normalize_number_words_line(text)
    else:
        return text, None

    raw_text = text if normalized_text != text else None
    return normalized_text, raw_text


def _validate_number_modes(number_word_mode: bool, number_digit_mode: bool) -> None:
    if number_word_mode and number_digit_mode:
        raise HTTPException(
            status_code=400,
            detail=(
                "number_word_mode and number_digit_mode cannot both be true. "
                "Please choose one mode."
            ),
        )


def _line_bounds_from_mask(ink_mask: np.ndarray) -> list[tuple[int, int]]:
    height, width = ink_mask.shape
    if height == 0 or width == 0:
        return []

    row_sum = ink_mask.sum(axis=1)
    row_threshold = max(2, int(width * 0.01))
    active_rows = row_sum >= row_threshold

    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            end = idx - 1
            if end - start + 1 >= 6:
                spans.append((start, end))
            start = None
    if start is not None:
        end = len(active_rows) - 1
        if end - start + 1 >= 6:
            spans.append((start, end))

    merged: list[tuple[int, int]] = []
    max_gap = 10
    for span in spans:
        if not merged:
            merged.append(span)
            continue
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = span
        if cur_start - prev_end <= max_gap:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append(span)
    return merged


def _split_lines(image: Image.Image) -> list[Image.Image]:
    gray = np.array(image.convert("L"))
    ink_mask = gray < 245
    if ink_mask.sum() < 20:
        return [image]

    spans = _line_bounds_from_mask(ink_mask)
    if len(spans) <= 1:
        return [image]

    lines: list[Image.Image] = []
    width = image.width
    for top, bottom in spans:
        top = max(0, top - 8)
        bottom = min(image.height - 1, bottom + 8)
        line_mask = ink_mask[top : bottom + 1, :]
        col_sum = line_mask.sum(axis=0)
        col_threshold = max(1, int((bottom - top + 1) * 0.02))
        active_cols = np.where(col_sum >= col_threshold)[0]
        if len(active_cols) == 0:
            continue

        left = max(0, int(active_cols[0]) - 8)
        right = min(width - 1, int(active_cols[-1]) + 8)
        if right - left < 5:
            continue

        line_ink_count = int(line_mask[:, left : right + 1].sum())
        if line_ink_count < 20:
            continue

        lines.append(image.crop((left, top, right + 1, bottom + 1)))

    if not lines:
        return [image]
    return lines


def _resolve_local_model_weights(model_name: str, weights_url: str) -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_weights = MODEL_CACHE_DIR / f"{model_name}.pth"

    if local_weights.exists() and local_weights.stat().st_size > 0:
        return local_weights

    tmp_path = local_weights.with_suffix(".pth.tmp")
    urlretrieve(weights_url, tmp_path)
    tmp_path.replace(local_weights)
    return local_weights


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

        weights_url = config["weights"]
        local_weights = _resolve_local_model_weights(model_name, weights_url)
        config["weights"] = str(local_weights)
        config["pretrain"] = str(local_weights)

        self.model_name = model_name
        self.device = device
        self.beamsearch = beamsearch
        self.predictor = Predictor(config)

    def _predict_one(
        self, image: Image.Image, return_prob: bool
    ) -> tuple[str, Optional[float]]:
        if return_prob:
            text, prob = self.predictor.predict(image, return_prob=True)
            return text, (None if prob is None else float(prob))
        text = self.predictor.predict(image, return_prob=False)
        return text, None

    def _predict_digit_best(
        self, image: Image.Image, return_prob: bool
    ) -> tuple[str, Optional[float], Optional[str]]:
        best_raw_text = ""
        best_probability = None
        best_score = float("-inf")

        for candidate_image in _preprocess_digit_variants(image):
            raw_text, probability = self._predict_one(
                image=candidate_image, return_prob=True
            )
            score = _score_digit_prediction(raw_text, probability)
            if score > best_score:
                best_raw_text = raw_text
                best_probability = probability
                best_score = score

        text, raw_text = _normalize_text_by_number_mode(
            text=best_raw_text,
            number_word_mode=False,
            number_digit_mode=True,
        )
        return text, (best_probability if return_prob else None), raw_text

    def recognize(
        self,
        image: Image.Image,
        return_prob: bool,
        multiline: bool,
        number_word_mode: bool,
        number_digit_mode: bool,
    ) -> OCRResponse:
        if not multiline:
            if number_digit_mode:
                text, probability, raw_text = self._predict_digit_best(
                    image=image, return_prob=return_prob
                )
            else:
                text, probability = self._predict_one(
                    image=image, return_prob=return_prob
                )
                text, raw_text = _normalize_text_by_number_mode(
                    text=text,
                    number_word_mode=number_word_mode,
                    number_digit_mode=number_digit_mode,
                )
            return OCRResponse(
                text=text,
                probability=probability,
                raw_text=raw_text,
                number_word_mode=number_word_mode,
                number_digit_mode=number_digit_mode,
                model=self.model_name,
                device=self.device,
            )

        if number_digit_mode:
            image = _preprocess_digit_image(image)

        line_images = _split_lines(image)
        if len(line_images) <= 1:
            if number_digit_mode:
                text, probability, raw_text = self._predict_digit_best(
                    image=image, return_prob=return_prob
                )
            else:
                text, probability = self._predict_one(
                    image=image, return_prob=return_prob
                )
                text, raw_text = _normalize_text_by_number_mode(
                    text=text,
                    number_word_mode=number_word_mode,
                    number_digit_mode=number_digit_mode,
                )
            return OCRResponse(
                text=text,
                probability=probability,
                lines=[text],
                line_probabilities=[probability],
                raw_text=raw_text,
                raw_lines=[raw_text] if raw_text else None,
                number_word_mode=number_word_mode,
                number_digit_mode=number_digit_mode,
                model=self.model_name,
                device=self.device,
            )

        lines: list[str] = []
        line_probs: list[Optional[float]] = []
        raw_line_candidates: list[str] = []
        for line_image in line_images:
            if number_digit_mode:
                line_text, line_prob, raw_line = self._predict_digit_best(
                    image=line_image, return_prob=return_prob
                )
                raw_line_candidates.append(raw_line if raw_line else line_text)
            else:
                line_text, line_prob = self._predict_one(
                    image=line_image, return_prob=return_prob
                )
            lines.append(line_text)
            line_probs.append(line_prob)

        raw_lines = None
        if number_digit_mode:
            if raw_line_candidates != lines:
                raw_lines = raw_line_candidates
        elif number_word_mode:
            original_lines = list(lines)
            lines = [
                _normalize_text_by_number_mode(
                    text=line,
                    number_word_mode=number_word_mode,
                    number_digit_mode=number_digit_mode,
                )[0]
                for line in lines
            ]
            if lines != original_lines:
                raw_lines = original_lines

        merged_text = "\n".join(lines)
        valid_probs = [p for p in line_probs if p is not None]
        probability = None if not valid_probs else sum(valid_probs) / len(valid_probs)

        return OCRResponse(
            text=merged_text,
            probability=probability,
            lines=lines,
            line_probabilities=line_probs,
            raw_text=("\n".join(raw_lines) if raw_lines else None),
            raw_lines=raw_lines,
            number_word_mode=number_word_mode,
            number_digit_mode=number_digit_mode,
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
    multiline: bool = True,
    number_word_mode: bool = False,
    number_digit_mode: bool = False,
) -> OCRResponse:
    _validate_number_modes(
        number_word_mode=number_word_mode, number_digit_mode=number_digit_mode
    )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    image = _open_image(content)
    service = get_ocr_service()
    return service.recognize(
        image=image,
        return_prob=return_prob,
        multiline=multiline,
        number_word_mode=number_word_mode,
        number_digit_mode=number_digit_mode,
    )


@app.post("/ocr/base64", response_model=OCRResponse)
def ocr_from_base64(payload: OCRBase64Request) -> OCRResponse:
    _validate_number_modes(
        number_word_mode=payload.number_word_mode,
        number_digit_mode=payload.number_digit_mode,
    )

    image_bytes = _decode_base64_image(payload.image_base64)
    image = _open_image(image_bytes)
    service = get_ocr_service()
    return service.recognize(
        image=image,
        return_prob=payload.return_prob,
        multiline=payload.multiline,
        number_word_mode=payload.number_word_mode,
        number_digit_mode=payload.number_digit_mode,
    )


@app.post("/debug/digit-variants/base64", response_model=DigitVariantResponse)
def digit_variants_from_base64(payload: OCRBase64Request) -> DigitVariantResponse:
    image_bytes = _decode_base64_image(payload.image_base64)
    image = _open_image(image_bytes)
    variants = _preprocess_digit_variants(image)
    return DigitVariantResponse(
        variants=[
            DigitVariantItem(
                index=index + 1,
                width=variant.width,
                height=variant.height,
                image_base64=_image_to_png_data_uri(variant),
            )
            for index, variant in enumerate(variants)
        ]
    )
