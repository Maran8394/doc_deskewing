import io
import json
import zipfile

from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader

from config import get_settings
from hough_transform_service import (
    BulkDeskewItem,
    ImageEncodingError,
    ImageProcessingError,
    ImageTooLargeError,
    InvalidImageError,
    encode_png,
    process_document_fast,
    process_documents_fast_bulk,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="Document Deskew API",
    description="Deskews document images using Hough line transform and returns an enhanced PNG.",
    version="1.0.0",
)

deskew_fast_response_docs = {
    200: {
        "description": "Deskewed PNG image.",
        "content": {"image/png": {}},
        "headers": {
            "X-Original-Angle": {
                "description": "Detected skew angle before correction.",
                "schema": {"type": "string", "example": "6.0000"},
            },
            "X-Rotated-Angle": {
                "description": "Detected skew angle after correction.",
                "schema": {"type": "string", "example": "0.5000"},
            },
            "X-Applied-Rotation": {
                "description": "Rotation angle applied to the uploaded image.",
                "schema": {"type": "string", "example": "6.0000"},
            },
            "X-Skew-Corrected": {
                "description": "Whether the API decided to deskew the image.",
                "schema": {"type": "string", "example": "true"},
            },
        },
    },
    401: {"description": "Invalid or missing API key."},
}

deskew_fast_bulk_response_docs = {
    200: {
        "description": "ZIP archive containing deskewed PNG files and results.json metadata.",
        "content": {"application/zip": {}},
        "headers": {
            "X-Processed-Count": {
                "description": "Number of files processed in the ZIP archive.",
                "schema": {"type": "string", "example": "2"},
            },
            "Content-Disposition": {
                "description": "Suggested ZIP filename.",
                "schema": {
                    "type": "string",
                    "example": 'attachment; filename="deskew_fast_bulk_results.zip"',
                },
            },
        },
    },
    401: {"description": "Invalid or missing API key."},
}


def validate_fast_api_key(api_key: str | None = Security(api_key_header)) -> None:
    settings = get_settings()
    if api_key != settings.fast_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/deskew-fast", responses=deskew_fast_response_docs)
async def deskew_document_fast(
    file: UploadFile = File(...),
    _: None = Depends(validate_fast_api_key),
) -> Response:
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    try:
        image_bytes = await file.read()
        result = process_document_fast(image_bytes)
        enhanced_image = encode_png(result.enhanced_image)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageEncodingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ImageProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while processing the image.",
        ) from exc
    finally:
        await file.close()

    headers = {
        "X-Original-Angle": f"{result.original_angle:.4f}",
        "X-Rotated-Angle": f"{result.rotated_angle:.4f}",
        "X-Applied-Rotation": f"{result.applied_rotation:.4f}",
        "X-Skew-Corrected": str(result.skewed).lower(),
    }
    return Response(content=enhanced_image, media_type="image/png", headers=headers)


def _build_bulk_zip(items: list[BulkDeskewItem]) -> bytes:
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


@app.post("/deskew-fast-bulk", responses=deskew_fast_bulk_response_docs)
async def deskew_documents_fast_bulk(
    files: list[UploadFile] = File(...),
    _: None = Depends(validate_fast_api_key),
) -> Response:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    uploads: list[tuple[str, bytes]] = []

    try:
        for file in files:
            content_type = file.content_type or ""
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only image uploads are supported: {file.filename or 'unnamed file'}.",
                )

            uploads.append((file.filename or "document", await file.read()))

        results = process_documents_fast_bulk(uploads)
        archive_bytes = _build_bulk_zip(results)
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImageEncodingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ImageProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while processing bulk images.",
        ) from exc
    finally:
        for file in files:
            await file.close()

    headers = {
        "Content-Disposition": 'attachment; filename="deskew_fast_bulk_results.zip"',
        "X-Processed-Count": str(len(results)),
    }
    return Response(content=archive_bytes, media_type="application/zip", headers=headers)


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
