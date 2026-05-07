from pathlib import Path

from doc_deskewing import deskew_images_bulk

compiled_documents = deskew_images_bulk(
    [("test_docs/doc1.pdf", Path("test_docs/doc1.pdf").read_bytes())],
    compile_pdf=True,
)

compiled_pdf = compiled_documents[0]
Path(compiled_pdf.output_filename).write_bytes(compiled_pdf.content_bytes)
