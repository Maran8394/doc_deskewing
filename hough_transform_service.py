from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.transform import hough_line, hough_line_peaks

MAX_IMAGE_BYTES: Final[int] = 20 * 1024 * 1024
DEFAULT_SKEW_THRESHOLD: Final[float] = 1.0
FAST_DETECTION_MAX_DIMENSION: Final[int] = 1200
FAST_SAFE_SKEW_LIMIT: Final[float] = 15.0
FAST_SEARCH_STEP: Final[float] = 0.5
FAST_MIN_SCORE_IMPROVEMENT_RATIO: Final[float] = 1.01
logger = logging.getLogger("uvicorn.error")


class ImageProcessingError(Exception):
    """Base error for image processing failures."""


class InvalidImageError(ImageProcessingError):
    """Raised when the input image cannot be decoded or is malformed."""


class ImageTooLargeError(ImageProcessingError):
    """Raised when the uploaded image exceeds the accepted size."""


class ImageEncodingError(ImageProcessingError):
    """Raised when the processed image cannot be encoded for response."""


@dataclass(slots=True)
class DeskewResult:
    original_angle: float
    rotated_angle: float
    applied_rotation: float
    skewed: bool
    corrected_image: np.ndarray
    enhanced_image: np.ndarray


@dataclass(slots=True)
class BulkDeskewItem:
    original_filename: str
    output_filename: str
    result: DeskewResult


def validate_image_size(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")
    if len(image_bytes) > max_bytes:
        raise ImageTooLargeError(
            f"Uploaded file exceeds the {max_bytes // (1024 * 1024)}MB limit."
        )


def decode_image(image_bytes: bytes) -> np.ndarray:
    validate_image_size(image_bytes)

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise InvalidImageError("Failed to decode uploaded image.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_to_content(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]


def enhance_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    return cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )


def binarize_image(rgb_image: np.ndarray) -> np.ndarray:
    if rgb_image.ndim != 3 or rgb_image.shape[2] not in (3, 4):
        raise InvalidImageError("Expected an RGB or RGBA image.")

    if rgb_image.shape[2] == 4:
        rgb_image = rgb_image[:, :, :3]

    grayscale = rgb2gray(rgb_image)
    threshold = threshold_otsu(grayscale)
    return grayscale < threshold


def find_edges(binary_image: np.ndarray) -> np.ndarray:
    return canny(binary_image)


def find_tilt_angle(image_edges: np.ndarray) -> float:
    hspace, theta, distances = hough_line(image_edges)

    if hspace.size == 0 or np.max(hspace) <= 0:
        return 0.0

    _, angles, _ = hough_line_peaks(
        hspace,
        theta,
        distances,
        threshold=0.3 * np.max(hspace),
        num_peaks=20,
    )

    if len(angles) == 0:
        return 0.0

    angle = float(np.median(np.rad2deg(angles)))
    return angle + 90 if angle < 0 else angle - 90


def is_skewed(angle: float, threshold: float = DEFAULT_SKEW_THRESHOLD) -> bool:
    return abs(angle) > threshold


def rotate_image(rgb_image: np.ndarray, angle: float) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        rgb_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return crop_to_content(rotated)


def resize_for_fast_detection(
    rgb_image: np.ndarray, max_dimension: int = FAST_DETECTION_MAX_DIMENSION
) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return rgb_image

    scale = max_dimension / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        rgb_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def normalize_angle_to_small_skew(
    angle: float, safe_limit: float = FAST_SAFE_SKEW_LIMIT
) -> float:
    candidates = (
        angle,
        angle - 90.0,
        angle + 90.0,
        angle - 180.0,
        angle + 180.0,
    )
    normalized = min(candidates, key=lambda candidate: abs(candidate))

    if abs(normalized) > safe_limit:
        return 0.0

    return normalized


def rotate_preview_image(rgb_image: np.ndarray, angle: float) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(
        rgb_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_tilt_angle_fast(rgb_image: np.ndarray) -> float:
    resized_image = resize_for_fast_detection(rgb_image)
    binary_image = binarize_image(resized_image)
    image_edges = find_edges(binary_image)
    raw_angle = find_tilt_angle(image_edges)
    normalized_angle = normalize_angle_to_small_skew(raw_angle)

    logger.info(
        "Fast angle detection: raw_angle=%.4f normalized_angle=%.4f",
        raw_angle,
        normalized_angle,
    )

    return normalized_angle


def horizontal_text_score(rgb_image: np.ndarray) -> float:
    binary_image = binarize_image(rgb_image)
    row_projection = np.sum(binary_image, axis=1)
    return float(np.std(row_projection))


def search_best_fast_rotation(
    rgb_image: np.ndarray,
    search_limit: float = FAST_SAFE_SKEW_LIMIT,
    step: float = FAST_SEARCH_STEP,
) -> tuple[float, float, float]:
    resized_image = resize_for_fast_detection(rgb_image)
    best_rotation = 0.0
    best_text_score = float("-inf")
    zero_rotation_score = horizontal_text_score(resized_image)

    for rotation in np.arange(-search_limit, search_limit + step / 2, step):
        rotated_preview = rotate_preview_image(resized_image, float(rotation))
        text_score = horizontal_text_score(rotated_preview)

        if text_score > best_text_score:
            best_rotation = float(rotation)
            best_text_score = text_score

    if best_text_score <= zero_rotation_score * FAST_MIN_SCORE_IMPROVEMENT_RATIO:
        best_rotation = 0.0

    rotated_preview = rotate_preview_image(resized_image, best_rotation)
    residual_angle = detect_tilt_angle_fast(rotated_preview)

    logger.info(
        "Fast rotation search: best_rotation=%.4f zero_score=%.4f best_score=%.4f residual_angle=%.4f",
        best_rotation,
        zero_rotation_score,
        best_text_score,
        residual_angle,
    )

    return best_rotation, zero_rotation_score, best_text_score


def process_document(image_bytes: bytes) -> DeskewResult:
    image = decode_image(image_bytes)
    binary_image = binarize_image(image)
    image_edges = find_edges(binary_image)
    original_angle = find_tilt_angle(image_edges)
    skewed = is_skewed(original_angle)
    applied_rotation = original_angle if skewed else 0.0
    corrected = (
        rotate_image(image, applied_rotation) if skewed else crop_to_content(image)
    )
    corrected_binary_image = binarize_image(corrected)
    corrected_edges = find_edges(corrected_binary_image)
    rotated_angle = find_tilt_angle(corrected_edges)

    enhanced = enhance_text(corrected)

    logger.info(
        "Deskew result: skewed=%s original_angle=%.4f applied_rotation=%.4f rotated_angle=%.4f",
        skewed,
        original_angle,
        applied_rotation,
        rotated_angle,
    )

    return DeskewResult(
        original_angle=original_angle,
        rotated_angle=rotated_angle,
        applied_rotation=applied_rotation,
        skewed=skewed,
        corrected_image=corrected,
        enhanced_image=enhanced,
    )


def process_document_fast(image_bytes: bytes) -> DeskewResult:
    image = decode_image(image_bytes)
    coarse_angle = detect_tilt_angle_fast(image)
    applied_rotation, _, _ = search_best_fast_rotation(image)
    original_angle = applied_rotation if applied_rotation != 0.0 else coarse_angle
    skewed = is_skewed(original_angle)

    if skewed:
        corrected = rotate_image(image, applied_rotation)
        rotated_angle = detect_tilt_angle_fast(corrected)
    else:
        applied_rotation = 0.0
        corrected = crop_to_content(image)
        rotated_angle = detect_tilt_angle_fast(corrected)

    enhanced = enhance_text(corrected)

    logger.info(
        "Fast deskew result: skewed=%s original_angle=%.4f applied_rotation=%.4f rotated_angle=%.4f",
        skewed,
        original_angle,
        applied_rotation,
        rotated_angle,
    )

    return DeskewResult(
        original_angle=original_angle,
        rotated_angle=rotated_angle,
        applied_rotation=applied_rotation,
        skewed=skewed,
        corrected_image=corrected,
        enhanced_image=enhanced,
    )


def encode_png(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        image_to_encode = image
    else:
        image_to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    success, encoded = cv2.imencode(".png", image_to_encode)
    if not success:
        raise ImageEncodingError("Failed to encode processed image as PNG.")

    return encoded.tobytes()


def build_output_filename(filename: str, suffix: str = "_deskewed.png") -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "document"
    return f"{safe_stem}{suffix}"


def process_documents_fast_bulk(
    files: list[tuple[str, bytes]],
) -> list[BulkDeskewItem]:
    results: list[BulkDeskewItem] = []

    for original_filename, image_bytes in files:
        result = process_document_fast(image_bytes)
        results.append(
            BulkDeskewItem(
                original_filename=original_filename,
                output_filename=build_output_filename(original_filename),
                result=result,
            )
        )

    return results
