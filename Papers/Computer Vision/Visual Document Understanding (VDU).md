Visual Document Understanding (VDU) is a field that encompasses the task of reading text and achieving a holistic understanding of document images (e.g., invoices, receipts, forms, business cards, academic papers).

## Key Tasks
- **Document Classification**: Categorizing documents into predefined types (e.g., letter, invoice, form). Benchmark: [[RVL-CDIP]]
- **Document Information Extraction**: Extracting structured key-value pairs from documents (e.g., total amount, date). Benchmark: [[CORD]]
- **Document Visual Question Answering**: Answering questions about a document image. Benchmark: [[DocVQA]]

## Approaches

### Traditional Pipeline (OCR-dependent)
1. An OCR engine extracts text from the document image
2. A downstream model processes the extracted text (and optionally layout information) for the target task

**Problems**: computational overhead, inflexibility across languages, error propagation from OCR mistakes into downstream tasks.

### End-to-End (OCR-free)
Models that directly process the document image without a separate OCR step, unifying text recognition and document understanding. Example: [[OCR-free Document Understanding Transformer (Donut)]]

## Related Papers
- [[OCR-free Document Understanding Transformer (Donut)]]
- [[Fast Vision Transformers with Hierarchical Attention (FasterViT)]]
