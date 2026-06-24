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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError


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
DIGIT_ALLOWED_SYMBOLS = set("+-x:=,*/.()")
MATH_OPERATOR_CHARS = set("+-x:=,")
MATH_OPERATOR_TOKEN_MAP = {
    "+": "+",
    "＋": "+",
    "plus": "+",
    "cong": "+",
    "cộng": "+",
    "daucong": "+",
    "dấu cộng": "+",
    "-": "-",
    "−": "-",
    "–": "-",
    "—": "-",
    "minus": "-",
    "tru": "-",
    "trừ": "-",
    "x": "x",
    "X": "x",
    "×": "x",
    "*": "x",
    "nhan": "x",
    "nhân": "x",
    ":": ":",
    "：": ":",
    "chia": ":",
    "=": "=",
    "＝": "=",
    ",": ",",
    "，": ",",
}


@lru_cache(maxsize=1)
def _cv2_module():
    import cv2

    return cv2


@lru_cache(maxsize=1)
def _torch_module():
    import torch

    return torch


class OCRResponse(BaseModel):
    text: str
    probability: Optional[float] = None
    lines: Optional[list[str]] = None
    line_probabilities: Optional[list[Optional[float]]] = None
    raw_text: Optional[str] = None
    raw_lines: Optional[list[str]] = None
    number_word_mode: bool = False
    number_digit_mode: bool = False
    math_symbol_mode: bool = False
    model: str
    device: str


class OCRBase64Request(BaseModel):
    image_base64: str = Field(..., description="Raw base64 or data URI")
    return_prob: bool = False
    multiline: bool = True
    number_word_mode: bool = False
    number_digit_mode: bool = False
    math_symbol_mode: bool = False


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
    torch = _torch_module()
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
    cv2 = _cv2_module()
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
    cv2 = _cv2_module()
    min_side = min(gray.shape[:2])
    if min_side < 3:
        return gray

    kernel_size = min(51, max(3, (min_side // 2) | 1))
    background = cv2.medianBlur(gray, kernel_size)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def _deskew_digit_image(binary_image: np.ndarray) -> np.ndarray:
    cv2 = _cv2_module()
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
    cv2 = _cv2_module()
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
    cv2 = _cv2_module()
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
    cv2 = _cv2_module()
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


def _normalize_math_operator_token(token: str) -> Optional[str]:
    value = token.strip()
    if not value:
        return None

    if value in MATH_OPERATOR_TOKEN_MAP:
        return MATH_OPERATOR_TOKEN_MAP[value]

    lowered = value.lower()
    if lowered in MATH_OPERATOR_TOKEN_MAP:
        return MATH_OPERATOR_TOKEN_MAP[lowered]

    plain = _strip_diacritics(lowered).replace(" ", "")
    return MATH_OPERATOR_TOKEN_MAP.get(plain)


def _normalize_number_digits_line(text: str) -> str:
    """
    Chi normalize ve chu so 0-9.

    Luu y: ham nay KHONG bat cac ky tu toan hoc nua.
    Logic + - x : = , da duoc tach sang math_symbol_mode.
    """
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return text.strip().lower()

    out: list[str] = []
    for token in tokens:
        if token.isspace():
            out.append(token)
            continue

        # Neu OCR doc ra tu/kien tuong ung voi toan tu, bo qua trong
        # number_digit_mode thuan. Muon bat cac ky tu nay thi bat math_symbol_mode.
        if _normalize_math_operator_token(token) is not None:
            continue

        if token.isdigit() or WORD_TOKEN_RE.fullmatch(token):
            out.append(_normalize_digit_token(token))
            continue

        # Trong number_digit_mode thuan, khong ep toan tu vao ket qua.
        # Chi giu cac confusion co the la chu so.
        mapped = DIGIT_OCR_CONFUSIONS.get(token.lower())
        if mapped is not None and mapped.isdigit():
            out.append(mapped)

    return "".join(out).strip()


def _normalize_math_symbols_line(text: str) -> str:
    """
    Normalize chuoi co chu so va ky tu toan hoc.

    Ham nay chi nen dung khi math_symbol_mode=True.
    Ho tro: 0-9 va + - x : = ,
    """
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return text.strip().lower()

    out: list[str] = []
    for token in tokens:
        if token.isspace():
            out.append(token)
            continue

        operator = _normalize_math_operator_token(token)
        if operator is not None:
            out.append(operator)
            continue

        if token.isdigit() or WORD_TOKEN_RE.fullmatch(token):
            out.append(_normalize_digit_token(token))
            continue

        mapped = DIGIT_OCR_CONFUSIONS.get(token.lower(), token)
        operator = _normalize_math_operator_token(mapped)
        if operator is not None:
            out.append(operator)
        else:
            out.append(mapped)

    allowed = set("0123456789+-x:=,")
    return "".join(ch for ch in "".join(out).strip() if ch in allowed)


def _score_digit_prediction(text: str, probability: Optional[float]) -> float:
    """
    Cham diem cho number_digit_mode thuan: uu tien chu so, khong thuong toan tu.
    """
    normalized = _normalize_number_digits_line(text)
    compact = re.sub(r"\s+", "", normalized)
    if not compact:
        return -1.0

    digit_count = sum(ch.isdigit() for ch in compact)
    bad_count = len(compact) - digit_count

    score = 0.0 if probability is None else probability
    score += min(digit_count, 24) * 0.03
    score -= bad_count * 0.12
    if digit_count == 0:
        score -= 0.5
    return score


def _normalize_text_by_number_mode(
    text: str,
    number_word_mode: bool,
    number_digit_mode: bool,
    math_symbol_mode: bool = False,
) -> tuple[str, Optional[str]]:
    if math_symbol_mode:
        normalized_text = _normalize_math_symbols_line(text)
    elif number_digit_mode:
        normalized_text = _normalize_number_digits_line(text)
    elif number_word_mode:
        normalized_text = _normalize_number_words_line(text)
    else:
        return text, None

    raw_text = text if normalized_text != text else None
    return normalized_text, raw_text


def _validate_number_modes(
    number_word_mode: bool,
    number_digit_mode: bool,
    math_symbol_mode: bool = False,
) -> None:
    if number_word_mode and (number_digit_mode or math_symbol_mode):
        raise HTTPException(
            status_code=400,
            detail=(
                "number_word_mode cannot be used with number_digit_mode or "
                "math_symbol_mode. Please choose text mode or numeric/symbol mode."
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




def _active_spans(active: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_active in enumerate(active):
        if bool(is_active) and start is None:
            start = idx
        elif not bool(is_active) and start is not None:
            end = idx - 1
            if end - start + 1 >= min_len:
                spans.append((start, end))
            start = None

    if start is not None:
        end = len(active) - 1
        if end - start + 1 >= min_len:
            spans.append((start, end))
    return spans


def _box_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return x1, y1, x2 - x1, y2 - y1


def _box_horizontal_overlap_ratio(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    ax, _, aw, _ = box_a
    bx, _, bw, _ = box_b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    return overlap / max(1, min(aw, bw))


def _box_vertical_overlap_ratio(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> float:
    _, ay, _, ah = box_a
    _, by, _, bh = box_b
    overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / max(1, min(ah, bh))


def _is_horizontal_stroke_box(box: tuple[int, int, int, int]) -> bool:
    _, _, width, height = box
    return width >= max(5, height * 1.8)


def _is_small_dot_box(box: tuple[int, int, int, int], line_height: int) -> bool:
    _, _, width, height = box
    # Dau ':' co the gom tu hai cham tron; khi anh chi co dau ':' thi line_height
    # cung rat nho, nen khong duoc dat nguong qua chat.
    max_dot_side = max(6, int(line_height * 1.4))
    return width <= max_dot_side and height <= max_dot_side and 0.45 <= width / max(height, 1) <= 2.2


def _is_equal_pair(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> bool:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if not (_is_horizontal_stroke_box(box_a) and _is_horizontal_stroke_box(box_b)):
        return False

    x_overlap = _box_horizontal_overlap_ratio(box_a, box_b)
    if x_overlap < 0.45:
        return False

    cy_a = ay + ah / 2
    cy_b = by + bh / 2
    vertical_gap = abs(cy_a - cy_b)
    if vertical_gap <= max(2, min(ah, bh) * 0.8):
        return False
    if vertical_gap > max(18, max(ah, bh) * 5):
        return False

    width_ratio = min(aw, bw) / max(aw, bw, 1)
    return width_ratio >= 0.45


def _is_colon_pair(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], line_height: int
) -> bool:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if not (_is_small_dot_box(box_a, line_height) and _is_small_dot_box(box_b, line_height)):
        return False

    cx_a = ax + aw / 2
    cx_b = bx + bw / 2
    if abs(cx_a - cx_b) > max(8, max(aw, bw) * 1.4):
        return False

    cy_a = ay + ah / 2
    cy_b = by + bh / 2
    vertical_gap = abs(cy_a - cy_b)
    if vertical_gap <= max(2, min(ah, bh) * 1.2):
        return False
    if vertical_gap > max(26, line_height * 0.75):
        return False

    union = _box_union(box_a, box_b)
    return union[3] >= union[2] * 1.2


def _is_cross_pair(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> bool:
    # Dung cho dau x viet roi hai net cheo khong dinh nhau.
    x_overlap = _box_horizontal_overlap_ratio(box_a, box_b)
    y_overlap = _box_vertical_overlap_ratio(box_a, box_b)
    if x_overlap < 0.35 or y_overlap < 0.35:
        return False

    union = _box_union(box_a, box_b)
    _, _, width, height = union
    ratio = width / max(height, 1)
    return 0.55 <= ratio <= 1.8


def _merge_operator_component_boxes(
    boxes: list[tuple[int, int, int, int]], line_height: int
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    used: set[int] = set()
    boxes = sorted(boxes, key=lambda box: (box[0], box[1]))

    for i, box in enumerate(boxes):
        if i in used:
            continue

        best_j = None
        best_priority = -1
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            other = boxes[j]

            # Chi xet cac component gan nhau de tranh gom nham chu so ke ben.
            union = _box_union(box, other)
            if union[2] > max(box[2], other[2]) * 2.8 and _box_horizontal_overlap_ratio(box, other) <= 0:
                continue

            if _is_equal_pair(box, other):
                best_j = j
                best_priority = 3
                break
            if _is_colon_pair(box, other, line_height) and best_priority < 2:
                best_j = j
                best_priority = 2
            elif _is_cross_pair(box, other) and best_priority < 1:
                best_j = j
                best_priority = 1

        if best_j is not None:
            merged.append(_box_union(box, boxes[best_j]))
            used.add(i)
            used.add(best_j)
        else:
            merged.append(box)
            used.add(i)

    merged.sort(key=lambda box: box[0])
    return merged


def _extract_symbol_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Tach anh phep tinh thanh cac box ky tu/ky hieu theo thu tu trai sang phai.
    Moi box co dang (x, y, width, height).
    """
    cv2 = _cv2_module()
    gray = np.array(image.convert("L"))
    if gray.size == 0:
        return []

    gray = _ensure_light_background(gray)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Noi nhe net bi dut. Khong dilate manh de tranh dinh so voi toan tu.
    kernel = np.ones((2, 2), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    image_h, image_w = binary.shape[:2]
    image_area = image_h * image_w
    min_area = max(6, int(image_area * 0.00008))
    raw_boxes: list[tuple[int, int, int, int]] = []

    for component_id in range(1, component_count):
        x, y, width, height, area = stats[component_id]
        if area < min_area:
            continue
        if width < 2 or height < 3:
            continue
        if width > image_w * 0.95 and height > image_h * 0.95:
            continue
        raw_boxes.append((int(x), int(y), int(width), int(height)))

    if not raw_boxes:
        return []

    line_height = max(box[3] for box in raw_boxes)
    merged_boxes = _merge_operator_component_boxes(raw_boxes, line_height=line_height)

    padded_boxes: list[tuple[int, int, int, int]] = []
    for x, y, width, height in merged_boxes:
        pad = max(2, min(6, int(max(width, height) * 0.18)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image_w, x + width + pad)
        y2 = min(image_h, y + height + pad)
        if x2 - x1 >= 2 and y2 - y1 >= 3:
            padded_boxes.append((x1, y1, x2 - x1, y2 - y1))

    padded_boxes.sort(key=lambda box: box[0])
    return padded_boxes


def _ink_mask_from_crop(crop: Image.Image) -> Optional[np.ndarray]:
    cv2 = _cv2_module()
    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return None

    gray = _ensure_light_background(gray)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    ink = binary > 0
    if int(ink.sum()) < 4:
        return None

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return ink[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _projection_scores(ink: np.ndarray) -> tuple[float, float, float]:
    height, width = ink.shape[:2]
    band_y = max(1, int(round(height * 0.16)))
    band_x = max(1, int(round(width * 0.16)))
    mid_y = height // 2
    mid_x = width // 2

    horizontal_band = ink[max(0, mid_y - band_y) : min(height, mid_y + band_y + 1), :]
    vertical_band = ink[:, max(0, mid_x - band_x) : min(width, mid_x + band_x + 1)]

    horizontal_score = horizontal_band.any(axis=0).sum() / max(width, 1)
    vertical_score = vertical_band.any(axis=1).sum() / max(height, 1)

    center = ink[
        max(0, mid_y - band_y) : min(height, mid_y + band_y + 1),
        max(0, mid_x - band_x) : min(width, mid_x + band_x + 1),
    ]
    center_density = center.sum() / max(center.size, 1)
    return float(horizontal_score), float(vertical_score), float(center_density)


def _corner_ink_ratio(ink: np.ndarray) -> float:
    height, width = ink.shape[:2]
    corner_h = max(1, height // 4)
    corner_w = max(1, width // 4)
    corners = [
        ink[:corner_h, :corner_w],
        ink[:corner_h, -corner_w:],
        ink[-corner_h:, :corner_w],
        ink[-corner_h:, -corner_w:],
    ]
    corner_ink = sum(int(corner.sum()) for corner in corners)
    return corner_ink / max(int(ink.sum()), 1)


def _diagonal_scores(ink: np.ndarray) -> tuple[float, float]:
    cv2 = _cv2_module()
    resized = cv2.resize(
        ink.astype(np.uint8), (32, 32), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    ys, xs = np.where(resized)
    if len(xs) == 0:
        return 0.0, 0.0

    diag_1 = (np.abs(ys - xs) <= 3).sum() / max(len(xs), 1)
    diag_2 = (np.abs(ys + xs - 31) <= 3).sum() / max(len(xs), 1)
    return float(diag_1), float(diag_2)



def _classify_minus_equal_by_geometry(
    crop: Image.Image,
    line_height: Optional[int] = None,
) -> Optional[str]:
    """
    Phan biet rieng '-' va '=' bang hinh hoc, uu tien '='.

    Fix quan trong:
    - Khong gop cac khoang trang nho giua 2 net cua '=' nua.
    - Neu row projection thay 2 cum net ngang tach nhau, tra '=' ngay.
    - Chi khi chi co 1 cum net ngang dai moi tra '-'.
    """
    cv2 = _cv2_module()

    gray = np.array(crop.convert("L"))
    if gray.size == 0:
        return None

    gray = _ensure_light_background(gray)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    ink = binary > 0
    if int(ink.sum()) < 8:
        return None

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None

    # Cat sat vung muc de phep do khong bi anh huong bo trang.
    binary = binary[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    ink = binary > 0

    height, width = ink.shape[:2]
    if height <= 0 or width <= 0:
        return None

    ratio = width / max(height, 1)

    # '-' va '=' phai co dang ngang ro. Dieu kien nay tranh bat nham so 2, 3, +, x.
    if ratio < 1.25:
        return None

    row_profile = ink.sum(axis=1) / max(width, 1)
    col_profile = ink.sum(axis=0) / max(height, 1)

    # Neu co net doc manh o giua thi co the la '+', khong xu ly o ham nay.
    vertical_density = float(col_profile.max(initial=0.0))
    if vertical_density >= 0.82 and ratio < 2.0:
        return None

    # ============================================================
    # Cach 1: Row projection. Day la cach quan trong nhat voi canvas cua ban.
    # Dau '=' co 2 cum hang co muc, tach nhau boi it nhat 1 hang trang.
    # Khong merge gap nho, vi gap nho chinh la dau hieu cua '=' viet bang net day.
    # ============================================================
    thresholds = (0.12, 0.16, 0.20, 0.24)

    for threshold in thresholds:
        active_rows = row_profile >= threshold
        spans = _active_spans(active_rows, min_len=1)

        # Loc span yeu/ngan. Moi net ngang phai co do phu theo chieu ngang du lon.
        strong_spans: list[tuple[int, int]] = []
        for start, end in spans:
            band_max = float(row_profile[start : end + 1].max(initial=0.0))
            span_height = end - start + 1
            if band_max >= max(0.18, threshold) and span_height <= max(14, int(height * 0.70)):
                strong_spans.append((start, end))

        if len(strong_spans) >= 2:
            top_start, top_end = strong_spans[0]
            bottom_start, bottom_end = strong_spans[-1]
            gap = bottom_start - top_end - 1

            # Voi net but day, gap co the chi 1 pixel. Van nen uu tien '='.
            if gap >= 1:
                return "="

    # ============================================================
    # Cach 2: Connected components. Bat truong hop 2 net cua '=' la 2 component.
    # ============================================================
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    horizontal_boxes: list[tuple[int, int, int, int, int]] = []
    for component_id in range(1, component_count):
        x, y, box_w, box_h, area = stats[component_id]
        if area < 8:
            continue

        component_ratio = box_w / max(box_h, 1)
        if component_ratio >= 1.8 and box_w >= max(8, int(width * 0.30)):
            horizontal_boxes.append((int(x), int(y), int(box_w), int(box_h), int(area)))

    if len(horizontal_boxes) >= 2:
        horizontal_boxes.sort(key=lambda box: box[1])
        top = horizontal_boxes[0]
        bottom = horizontal_boxes[-1]

        top_x, top_y, top_w, top_h, _ = top
        bot_x, bot_y, bot_w, bot_h, _ = bottom

        overlap_x = max(0, min(top_x + top_w, bot_x + bot_w) - max(top_x, bot_x))
        overlap_ratio = overlap_x / max(1, min(top_w, bot_w))
        vertical_gap = bot_y - (top_y + top_h)

        if overlap_ratio >= 0.25 and vertical_gap >= 1:
            return "="

    # ============================================================
    # Chi khi khong co dau hieu 2 net, moi ket luan '-'.
    # ============================================================
    active_rows = row_profile >= 0.20
    spans = _active_spans(active_rows, min_len=1)
    strong_spans: list[tuple[int, int]] = []
    for start, end in spans:
        band_max = float(row_profile[start : end + 1].max(initial=0.0))
        if band_max >= 0.25:
            strong_spans.append((start, end))

    if len(strong_spans) == 1:
        stroke_start, stroke_end = strong_spans[0]
        stroke_height = stroke_end - stroke_start + 1

        # '-' thuong la 1 net ngang dai, crop thap hon so voi chieu rong.
        looks_flat = ratio >= 1.65
        stroke_not_too_tall = stroke_height <= max(18, int(height * 0.88))

        size_ok = True
        if line_height is not None and line_height > height:
            size_ok = height <= max(18, int(line_height * 0.85))

        if looks_flat and stroke_not_too_tall and size_ok:
            return "-"

    if len(horizontal_boxes) == 1 and ratio >= 1.65:
        return "-"

    return None


def _classify_math_operator_crop(crop: Image.Image, line_height: Optional[int] = None) -> Optional[str]:
    """
    Nhan dien rieng cac toan tu + - x : = , bang hinh hoc.
    Ham nay chi duoc dung trong number_digit_mode, khong anh huong luong viet chu.
    """
    ink = _ink_mask_from_crop(crop)
    if ink is None:
        return None

    height, width = ink.shape[:2]
    ratio = width / max(height, 1)
    ink_count = int(ink.sum())
    density = ink_count / max(width * height, 1)
    horizontal_score, vertical_score, center_density = _projection_scores(ink)
    corner_ratio = _corner_ink_ratio(ink)

    row_profile = ink.sum(axis=1) / max(width, 1)
    col_profile = ink.sum(axis=0) / max(height, 1)
    row_spans = _active_spans(row_profile >= 0.25, min_len=1)
    col_spans = _active_spans(col_profile >= 0.25, min_len=1)
    diag_1, diag_2 = _diagonal_scores(ink)

    # Dau bang: hai net ngang tach nhau.
    if ratio >= 1.15 and len(row_spans) >= 2:
        top_span = row_spans[0]
        bottom_span = row_spans[-1]
        gap = bottom_span[0] - top_span[1]
        row_span_lengths = [end - start + 1 for start, end in row_spans]
        if gap >= 2 and max(row_span_lengths) <= max(8, int(height * 0.45)):
            if len(col_spans) <= 2 or col_profile.max(initial=0) >= 0.35:
                return "="

    # Dau tru: mot net ngang dai.
    if ratio >= 1.8 and len(row_spans) <= 2 and horizontal_score >= 0.55:
        if height <= max(8, width * 0.45):
            return "-"

    # Dau nhan x: hai duong cheo cat nhau. Dat truoc dau cong vi x cung co muc o gan tam.
    if 0.45 <= ratio <= 2.2:
        if diag_1 >= 0.25 and diag_2 >= 0.25 and center_density >= 0.08:
            if horizontal_score < 0.85 and vertical_score < 0.85:
                return "x"

    # Dau hai cham: hai cum muc nho theo chieu doc.
    if ratio <= 1.25 and len(row_spans) >= 2 and len(col_spans) <= 2:
        top_span = row_spans[0]
        bottom_span = row_spans[-1]
        gap = bottom_span[0] - top_span[1]
        row_span_lengths = [end - start + 1 for start, end in row_spans]
        if gap >= 2 and max(row_span_lengths) <= max(6, int(height * 0.45)):
            if horizontal_score <= 0.75 and vertical_score >= 0.30 and density <= 0.75:
                return ":"

    # Dau phay: chi bat khi no nho hon chieu cao ky tu chinh trong dong.
    # Lam nhu vay de tranh nham chu so 1 thanh dau phay.
    if ratio <= 0.85 and height >= 4 and density <= 0.78:
        if line_height is not None and height <= max(8, int(line_height * 0.72)):
            return ","
        top_ink = ink[: max(1, height // 2), :].sum()
        bottom_ink = ink[max(1, height // 2) :, :].sum()
        if line_height is None and bottom_ink >= top_ink * 1.25:
            return ","

    # Dau cong: co net ngang giua va net doc giua, bon goc kha trong.
    if 0.60 <= ratio <= 2.2 and density <= 0.62:
        if horizontal_score >= 0.38 and vertical_score >= 0.38 and center_density >= 0.12:
            if corner_ratio <= 0.42 and not (diag_1 >= 0.25 and diag_2 >= 0.25):
                return "+"

    return None


def _normalize_math_component_text(text: str) -> str:
    compact = re.sub(r"\s+", "", text.strip())
    if not compact:
        return ""

    operator = _normalize_math_operator_token(compact)
    if operator is not None:
        return operator

    normalized = _normalize_number_digits_line(compact)
    normalized = re.sub(r"\s+", "", normalized)

    # Giu lai cac ky tu hop le cua mode uu tien so/toan tu.
    allowed = set("0123456789+-x:=,")
    filtered = "".join(ch for ch in normalized if ch in allowed)
    return filtered if filtered else normalized



# =========================
# Math symbol CNN inference
# =========================
# Dung cho math_symbol_mode=True.
# Model nay nhan dien 16 class: 0-9 va + - x : = ,
# khong dung VietOCR de doan tung ky tu nua.

def _math_symbol_weights_path() -> Path:
    raw = os.getenv("MATH_SYMBOL_WEIGHTS")
    if raw:
        return Path(raw)
    return APP_DIR / "weights" / "math_symbol_cnn.pt"


def _resolve_math_symbol_device() -> str:
    forced_device = os.getenv("MATH_SYMBOL_DEVICE")
    if forced_device:
        return forced_device
    return _resolve_device()


def _make_small_symbol_cnn(num_classes: int):
    torch = _torch_module()
    nn = torch.nn

    class SmallSymbolCNN(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 192, 3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(192, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SmallSymbolCNN(num_classes=num_classes)


def _normalize_symbol_image_for_model(
    image: Image.Image, img_size: int = 64, pad: int = 8
) -> Image.Image:
    gray = image.convert("L")
    arr = np.array(gray)
    if arr.size == 0:
        return Image.new("L", (img_size, img_size), 255)

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if float(np.median(border)) < 127.0:
        gray = ImageOps.invert(gray)
        arr = np.array(gray)

    ink = arr < 245
    if int(ink.sum()) > 4:
        ys, xs = np.where(ink)
        gray = gray.crop(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        )

    width, height = gray.size
    side = max(width, height) + 2 * pad
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((img_size, img_size), Image.Resampling.LANCZOS)


class MathSymbolClassifier:
    def __init__(self, weights_path: Path, device: str) -> None:
        torch = _torch_module()
        self.torch = torch
        self.device = device
        self.weights_path = weights_path

        if not weights_path.exists():
            raise RuntimeError(
                "Math symbol model not found. Train it first or set "
                f"MATH_SYMBOL_WEIGHTS. Missing file: {weights_path}"
            )

        checkpoint = torch.load(str(weights_path), map_location=device)
        self.class_names = checkpoint["class_names"]
        self.class_to_symbol = checkpoint["class_to_symbol"]
        self.img_size = int(checkpoint.get("img_size", 64))

        self.model = _make_small_symbol_cnn(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(device)
        self.model.eval()

    def predict(self, image: Image.Image) -> tuple[str, float]:
        torch = self.torch
        normalized = _normalize_symbol_image_for_model(
            image, img_size=self.img_size
        )
        arr = np.array(normalized).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]

        index = int(torch.argmax(probabilities).item())
        class_name = self.class_names[index]
        symbol = self.class_to_symbol[class_name]
        confidence = float(probabilities[index].item())
        return symbol, confidence


@lru_cache(maxsize=1)
def get_math_symbol_classifier() -> MathSymbolClassifier:
    return MathSymbolClassifier(
        weights_path=_math_symbol_weights_path(),
        device=_resolve_math_symbol_device(),
    )

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
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

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

    def _predict_math_components(
        self, image: Image.Image, return_prob: bool
    ) -> tuple[str, Optional[float], Optional[str]]:
        """
        math_symbol_mode=True:
        - OpenCV tach tung box ky tu/ky hieu.
        - CNN 16 class nhan dien moi box: 0-9 va + - x : = ,
        - Khong dung VietOCR trong nhanh nay nua.
        """
        try:
            classifier = get_math_symbol_classifier()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        boxes = _extract_symbol_boxes(image)
        if not boxes:
            symbol, confidence = classifier.predict(image)
            return symbol, (confidence if return_prob else None), f"{symbol}:{confidence:.4f}"

        parts: list[str] = []
        raw_parts: list[str] = []
        probabilities: list[float] = []

        line_height = max(height for _, _, _, height in boxes)

        for x, y, width, height in boxes:
            crop = image.crop((x, y, x + width, y + height))

            # '-' và '=' rất dễ bị CNN nhầm nhau. Hậu kiểm bằng hình học trước:
            # một nét ngang dài => '-', hai nét ngang tách nhau => '='.
            geometry_symbol = _classify_minus_equal_by_geometry(
                crop, line_height=line_height
            )
            if geometry_symbol is not None:
                symbol = geometry_symbol
                confidence = 1.0
            else:
                symbol, confidence = classifier.predict(crop)

            parts.append(symbol)
            raw_parts.append(f"{symbol}:{confidence:.4f}")
            probabilities.append(confidence)

        merged_text = "".join(parts)
        probability = None
        if return_prob and probabilities:
            probability = sum(probabilities) / len(probabilities)

        raw_text = " ".join(raw_parts)
        return merged_text, probability, raw_text

    def _active_response_model(self, math_symbol_mode: bool) -> str:
        if math_symbol_mode:
            return "math_symbol_cnn"
        return self.model_name

    def _active_response_device(self, math_symbol_mode: bool) -> str:
        if not math_symbol_mode:
            return self.device
        try:
            return get_math_symbol_classifier().device
        except Exception:  # noqa: BLE001
            return _resolve_math_symbol_device()


    def recognize(
        self,
        image: Image.Image,
        return_prob: bool,
        multiline: bool,
        number_word_mode: bool,
        number_digit_mode: bool,
        math_symbol_mode: bool = False,
    ) -> OCRResponse:
        if not multiline:
            if math_symbol_mode:
                text, probability, raw_text = self._predict_math_components(
                    image=image, return_prob=return_prob
                )
            elif number_digit_mode:
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
                    math_symbol_mode=math_symbol_mode,
                )
            return OCRResponse(
                text=text,
                probability=probability,
                raw_text=raw_text,
                number_word_mode=number_word_mode,
                number_digit_mode=number_digit_mode,
                math_symbol_mode=math_symbol_mode,
                model=self._active_response_model(math_symbol_mode),
                device=self._active_response_device(math_symbol_mode),
            )

        line_images = _split_lines(image)
        if len(line_images) <= 1:
            if math_symbol_mode:
                text, probability, raw_text = self._predict_math_components(
                    image=image, return_prob=return_prob
                )
            elif number_digit_mode:
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
                    math_symbol_mode=math_symbol_mode,
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
                math_symbol_mode=math_symbol_mode,
                model=self._active_response_model(math_symbol_mode),
                device=self._active_response_device(math_symbol_mode),
            )

        lines: list[str] = []
        line_probs: list[Optional[float]] = []
        raw_line_candidates: list[str] = []
        for line_image in line_images:
            if math_symbol_mode:
                line_text, line_prob, raw_line = self._predict_math_components(
                    image=line_image, return_prob=return_prob
                )
                raw_line_candidates.append(raw_line if raw_line else line_text)
            elif number_digit_mode:
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
        if math_symbol_mode or number_digit_mode:
            if raw_line_candidates != lines:
                raw_lines = raw_line_candidates
        elif number_word_mode:
            original_lines = list(lines)
            lines = [
                _normalize_text_by_number_mode(
                    text=line,
                    number_word_mode=number_word_mode,
                    number_digit_mode=number_digit_mode,
                    math_symbol_mode=math_symbol_mode,
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
            math_symbol_mode=math_symbol_mode,
            model=self._active_response_model(math_symbol_mode),
            device=self._active_response_device(math_symbol_mode),
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": os.getenv("VIETOCR_MODEL", "vgg_transformer"),
        "device": os.getenv("VIETOCR_DEVICE", "auto"),
        "beamsearch": _env_bool("VIETOCR_BEAMSEARCH", True),
        "service_loaded": get_ocr_service.cache_info().currsize > 0,
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
    math_symbol_mode: bool = False,
) -> OCRResponse:
    _validate_number_modes(
        number_word_mode=number_word_mode,
        number_digit_mode=number_digit_mode,
        math_symbol_mode=math_symbol_mode,
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
        math_symbol_mode=math_symbol_mode,
    )


@app.post("/ocr/base64", response_model=OCRResponse)
def ocr_from_base64(payload: OCRBase64Request) -> OCRResponse:
    _validate_number_modes(
        number_word_mode=payload.number_word_mode,
        number_digit_mode=payload.number_digit_mode,
        math_symbol_mode=payload.math_symbol_mode,
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
        math_symbol_mode=payload.math_symbol_mode,
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
