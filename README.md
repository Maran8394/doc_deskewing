# doc-deskewing

Internal pip package for document deskewing.

## Install From Git

```bash
pip install "git@github.com:Maran8394/doc_deskewing.git"
```

Or pin a branch / tag:

```bash
pip install "git@github.com:Maran8394/doc_deskewing.git@main"
pip install "git@github.com:Maran8394/doc_deskewing.git@v1.0.0"
```

For local development:

```bash
pip install .
```

## Import And Use In Python

```python
from pathlib import Path

from doc_deskewing import deskew_image, encode_png

image_bytes = Path("invoice.jpg").read_bytes()
result = deskew_image(image_bytes)
Path("invoice_deskewed.png").write_bytes(encode_png(result.enhanced_image))
```

`deskew_image` accepts:

- An image file as bytes
- A single-page PDF as bytes

If the detected angle is within the skew threshold (default is **1.0 degree**), the package skips rotation and text enhancement, returning the original image to preserve quality.

### DeskewResult Attributes

When you call `deskew_image`, it returns a `DeskewResult` object with the following fields:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `original_angle` | `float` | The detected skew angle of the input document. |
| `rotated_angle` | `float` | The detected angle after correction (should be near 0.0). |
| `applied_rotation` | `float` | The actual rotation applied (0.0 if not skewed). |
| `skewed` | `bool` | `True` if the `abs(original_angle)` exceeded the 1.0° threshold. |
| `corrected_image` | `ndarray` | The image after rotation and cropping (original if not skewed). |
| `enhanced_image` | `ndarray` | The binarized/cleaned image ready for OCR (original if not skewed). |

Bulk usage:

```python
from pathlib import Path

from doc_deskewing import deskew_images_bulk, encode_png

files = [
    ("invoice_1.jpg", Path("invoice_1.jpg").read_bytes()),
    ("statement.pdf", Path("statement.pdf").read_bytes()),
]

results = deskew_images_bulk(files)

for item in results:
    Path(item.output_filename).write_bytes(encode_png(item.result.enhanced_image))
```

Available imports:

```python
from doc_deskewing import (
    deskew_image,
    deskew_images_bulk,
    encode_png,
    DeskewResult,
    BulkDeskewItem,
    CompiledDocument,
)
```

`deskew_images_bulk` accepts multiple files. Each file can be:

- An image
- A single-page PDF
- A multi-page PDF

For multi-page PDFs, the method returns one `BulkDeskewItem` per page.

If you pass `compile_pdf=True`, the method accepts PDF inputs only and returns one `CompiledDocument` per input PDF.

## Expected Inputs

1. Read the source file into bytes.
2. Pass the bytes into `deskew_image(...)` for one file or `deskew_images_bulk(...)` for many files.
3. Save `result.enhanced_image` using `encode_png(...)`.

Example with a single-page PDF:

```python
from pathlib import Path

from doc_deskewing import deskew_image, encode_png

pdf_bytes = Path("page.pdf").read_bytes()
result = deskew_image(pdf_bytes)
Path("page_deskewed.png").write_bytes(encode_png(result.enhanced_image))
```

Example with a multi-page PDF:

```python
from pathlib import Path

from doc_deskewing import deskew_images_bulk, encode_png

results = deskew_images_bulk([
    ("report.pdf", Path("report.pdf").read_bytes()),
])

for item in results:
    Path(item.output_filename).write_bytes(encode_png(item.result.enhanced_image))
```

Example with a multi-page PDF compiled back into one PDF:

```python
from pathlib import Path

from doc_deskewing import deskew_images_bulk

compiled_documents = deskew_images_bulk(
    [("report.pdf", Path("report.pdf").read_bytes())],
    compile_pdf=True,
)

compiled_pdf = compiled_documents[0]
Path(compiled_pdf.output_filename).write_bytes(compiled_pdf.content_bytes)
```

When `compile_pdf=True`:

- input files must be PDFs
- output items are `CompiledDocument`
- each `CompiledDocument` still includes `items` with the per-page `BulkDeskewItem` results

## Publish internally

Typical flow:

1. Build the wheel with `python -m build`
2. Upload `dist/*.whl` to your internal PyPI or artifact registry
3. Install in other services with `pip install doc-deskewing==1.0.0`
