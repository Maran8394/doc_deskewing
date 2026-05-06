# Deskew Fast API Docs

Base URL:
`http://127.0.0.1:8000`

Protected endpoints:
- `POST /deskew-fast`
- `POST /deskew-fast-bulk`

Authentication:
- Header: `X-API-Key`
- Value source: `.env` -> `DESKEW_FAST_API_KEY`

Swagger UI:
- Open `http://127.0.0.1:8000/docs`
- Click `Authorize`
- Enter the `.env` API key for `X-API-Key`

Single file request:
```bash
curl -X POST "http://127.0.0.1:8000/deskew-fast" \
  -H "X-API-Key: NT3sA7V0N5zBgDHcPZK5Yz43IVexoV68VIV3RjI_VDI" \
  -F "file=@invoice.jpg" \
  -o invoice_deskewed.png -D -
```

Bulk request:
```bash
curl -X POST "http://127.0.0.1:8000/deskew-fast-bulk" \
  -H "X-API-Key: NT3sA7V0N5zBgDHcPZK5Yz43IVexoV68VIV3RjI_VDI" \
  -F "files=@invoice_1.jpg" \
  -F "files=@invoice_2.jpg" \
  -o deskew_fast_bulk_results.zip -D -
```

Single-file response:
- Body: processed PNG
- Headers:
  - `X-Original-Angle`
  - `X-Rotated-Angle`
  - `X-Applied-Rotation`
  - `X-Skew-Corrected`

Bulk response:
- Body: ZIP archive
- ZIP contents:
  - processed PNG per input file
  - `results.json` with per-file metadata
