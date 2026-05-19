> [!quote] Information
> * calendar Date 2020
> * paper Paper [Link](https://arxiv.org/abs/2007.00398)
> * ? Description:
>   Document Visual Question Answering dataset

## Overview

DocVQA is a dataset for Visual Question Answering on document images, introduced by Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. It was presented at WACV 2021. The dataset requires models to answer questions about the content of document images, combining challenges from both visual question answering and document image analysis.

The document images are sourced from the UCSF Industry Documents Library and cover a wide variety of document types including reports, invoices, forms, and letters. Questions and answers were manually annotated, and the answer is always a single span of text extracted from the given document image. This extractive setup makes the task particularly challenging as models must both recognize text and understand the spatial layout and structure of the documents.

The dataset includes 9 question types that reflect the kind of data where the question is grounded, such as table/list-based, form-based, layout-based, and handwritten text questions. Human performance on the dataset reaches 94.36% accuracy, while existing models show a significant performance gap, especially on questions that require understanding the structure of the document.

## Statistics

- **Total questions:** 50,000
- **Total document images:** 12,000+
- **Train:** 39,463 questions / 10,194 images
- **Validation:** 5,349 questions / 1,286 images
- **Test:** 5,188 questions / 1,287 images
- **Question types:** 9 categories
- **Average answer length:** 2.17 tokens
- **Unique answers:** 63.2%
- **Human performance:** 94.36% accuracy

## Used in

[[OCR-free Document Understanding Transformer (Donut)]]
