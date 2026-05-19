> [!quote] Information
> * calendar Date 2019
> * paper Paper [Link](https://openreview.net/pdf?id=SJl3z659UH)
> * ? Description:
>   Consolidated Receipt Dataset for Post-OCR Parsing

## Overview

CORD (Consolidated Receipt Dataset) is the first publicly available dataset that includes both box-level text and parsing class annotations for receipt understanding. It was introduced at the Document Intelligence Workshop at NeurIPS 2019 by Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee from Clova AI Research (NAVER Corp).

The dataset was created to address the lack of unified datasets for post-OCR parsing, where OCR and semantic parsing had traditionally been studied as separate tasks. CORD bridges this gap by providing receipt images together with OCR annotations and multi-level semantic labels, enabling research on integrated document understanding pipelines.

The receipts are Indonesian receipts collected from shops and restaurants. The dataset uses a multi-hierarchy label structure with 30 primary semantic categories organized into five superclasses (menu, void menu, subtotal, void total, and total), with 42 subclass labels in total. It also includes supplementary attributes such as line grouping, region of interest, cut lines, and key-value flags.

## Statistics

- **Total images:** 11,000+ Indonesian receipts (with OCR annotations)
- **Train:** 800 samples
- **Validation:** 100 samples
- **Test:** 100 samples
- **Semantic categories:** 30 primary classes across 5 superclasses
- **Subclass labels:** 42 total
- **License:** Creative Commons Attribution 4.0

## Used in

[[OCR-free Document Understanding Transformer (Donut)]]
