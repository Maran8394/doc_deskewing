from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Final

import cv2
import fitz
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
PDF_RENDER_DPI: Final[int] = 200
logger = logging.getLogger("doc_deskewing")


class ImageProcessingError(Exception):
    """Base error for image processing failures."""


class InvalidImageError(ImageProcessingError):
    """Raised when the input image cannot be decoded or is malformed."""


class ImageTooLargeError(ImageProcessingError):
    """Raised when the uploaded image exceeds the accepted size."""


class ImageEncodingError(ImageProcessingError):
    """Raised when the processed image cannot be encoded for response."""


class InvalidDocumentError(ImageProcessingError):
    """Raised when the input document type is unsupported or malformed."""


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


@dataclass(slots=True)
class CompiledDocument:
    original_filename: str
    output_filename: str
    content_bytes: bytes
    media_type: str
    items: list[BulkDeskewItem]


def validate_image_size(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    if not image_bytes:
        raise InvalidImageError("Uploaded file is empty.")
    if len(image_bytes) > max_bytes:
        raise ImageTooLargeError(
            f"Uploaded file exceeds the {max_bytes // (1024 * 1024)}MB limit."
        )


def is_pdf_bytes(document_bytes: bytes) -> bool:
    return document_bytes[:5] == b"%PDF-"


def decode_pdf_page(pdf_bytes: bytes, page_index: int = 0, dpi: int = PDF_RENDER_DPI) -> np.ndarray:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError("Failed to open PDF document.") from exc

    with pdf:
        if pdf.page_count == 0:
            raise InvalidDocumentError("PDF document has no pages.")
        if page_index < 0 or page_index >= pdf.page_count:
            raise InvalidDocumentError(f"PDF page index {page_index} is out of range.")

        page = pdf.load_page(page_index)
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

    buffer = np.frombuffer(pixmap.samples, dtype=np.uint8)
    image = buffer.reshape(pixmap.height, pixmap.width, pixmap.n)
    if pixmap.n == 4:
        image = image[:, :, :3]
    return image.copy()


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError("Failed to open PDF document.") from exc

    with pdf:
        if pdf.page_count == 0:
            raise InvalidDocumentError("PDF document has no pages.")
        return pdf.page_count


def decode_pdf_pages(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> list[np.ndarray]:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError("Failed to open PDF document.") from exc

    pages: list[np.ndarray] = []
    with pdf:
        if pdf.page_count == 0:
            raise InvalidDocumentError("PDF document has no pages.")

        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            buffer = np.frombuffer(pixmap.samples, dtype=np.uint8)
            image = buffer.reshape(pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n == 4:
                image = image[:, :, :3]
            pages.append(image.copy())

    return pages


def decode_image(image_bytes: bytes) -> np.ndarray:
    validate_image_size(image_bytes)

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise InvalidImageError("Failed to decode uploaded image.")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_document(document_bytes: bytes) -> np.ndarray:
    validate_image_size(document_bytes)
    if is_pdf_bytes(document_bytes):
        return decode_pdf_page(document_bytes, page_index=0)
    return decode_image(document_bytes)


def decode_document_pages(document_bytes: bytes) -> list[np.ndarray]:
    validate_image_size(document_bytes)
    if is_pdf_bytes(document_bytes):
        return decode_pdf_pages(document_bytes)
    return [decode_image(document_bytes)]


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


def deskew_image_bytes(image: np.ndarray) -> DeskewResult:
    coarse_angle = detect_tilt_angle_fast(image)
    applied_rotation, _, _ = search_best_fast_rotation(image)
    original_angle = applied_rotation if applied_rotation != 0.0 else coarse_angle
    skewed = is_skewed(original_angle)
    applied_rotation = applied_rotation if skewed else 0.0

    if skewed:
        corrected = rotate_image(image, applied_rotation)
        rotated_angle = detect_tilt_angle_fast(corrected)
        enhanced = enhance_text(corrected)
    else:
        corrected = image
        rotated_angle = original_angle
        enhanced = image

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


def deskew_image(document_bytes: bytes) -> DeskewResult:
    if is_pdf_bytes(document_bytes) and get_pdf_page_count(document_bytes) != 1:
        raise InvalidDocumentError(
            "deskew_image accepts image files or single-page PDFs only."
        )

    image = decode_document(document_bytes)
    return deskew_image_bytes(image)


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


def build_output_filename_for_page(
    filename: str,
    page_number: int,
    suffix: str = "_page_{page_number}_deskewed.png",
) -> str:
    return build_output_filename(filename, suffix=suffix.format(page_number=page_number))


def build_output_pdf_filename(filename: str, suffix: str = "_deskewed.pdf") -> str:
    return build_output_filename(filename, suffix=suffix)


def build_pdf_document(items: list[BulkDeskewItem]) -> bytes:
    pdf = fitz.open()

    try:
        for item in items:
            image_bytes = encode_png(item.result.enhanced_image)
            image_doc = fitz.open(stream=image_bytes, filetype="png")
            rect = image_doc[0].rect
            page = pdf.new_page(width=rect.width, height=rect.height)
            page.insert_image(rect, stream=image_bytes)
            image_doc.close()

        return pdf.tobytes()
    finally:
        pdf.close()


def deskew_images_bulk(
    files: list[tuple[str, bytes]],
    compile_pdf: bool = False,
) -> list[BulkDeskewItem] | list[CompiledDocument]:
    if compile_pdf:
        compiled_documents: list[CompiledDocument] = []

        for original_filename, document_bytes in files:
            if not is_pdf_bytes(document_bytes):
                raise InvalidDocumentError(
                    "compile_pdf=True is supported for PDF inputs only."
                )

            pages = decode_document_pages(document_bytes)
            items: list[BulkDeskewItem] = []

            for page_index, page_image in enumerate(pages, start=1):
                result = deskew_image_bytes(page_image)
                items.append(
                    BulkDeskewItem(
                        original_filename=original_filename,
                        output_filename=build_output_filename_for_page(
                            original_filename,
                            page_index,
                        ),
                        result=result,
                    )
                )

            compiled_documents.append(
                CompiledDocument(
                    original_filename=original_filename,
                    output_filename=build_output_pdf_filename(original_filename),
                    content_bytes=build_pdf_document(items),
                    media_type="application/pdf",
                    items=items,
                )
            )

        return compiled_documents

    results: list[BulkDeskewItem] = []

    for original_filename, document_bytes in files:
        pages = decode_document_pages(document_bytes)
        is_multi_page_pdf = is_pdf_bytes(document_bytes) and len(pages) > 1

        for page_index, page_image in enumerate(pages, start=1):
            result = deskew_image_bytes(page_image)
            output_filename = (
                build_output_filename_for_page(original_filename, page_index)
                if is_multi_page_pdf
                else build_output_filename(original_filename)
            )
            results.append(
                BulkDeskewItem(
                    original_filename=original_filename,
                    output_filename=output_filename,
                    result=result,
                )
            )

    return results


def build_bulk_zip(items: list[BulkDeskewItem]) -> bytes:
    metadata: list[dict[str, str | float | bool]] = []
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            archive.writestr(item.output_filename, encode_png(item.result.enhanced_image))
            metadata.append(
                {
                    "input_filename": item.original_filename,
                    "output_filename": item.output_filename,
                    "original_angle": round(item.result.original_angle, 4),
                    "rotated_angle": round(item.result.rotated_angle, 4),
                    "applied_rotation": round(item.result.applied_rotation, 4),
                    "skew_corrected": item.result.skewed,
                }
            )

        archive.writestr("results.json", json.dumps(metadata, indent=2))

    return buffer.getvalue()
