> [!quote] Information
> * calendar Date 2015
> * paper Paper [Link](https://arxiv.org/abs/1502.07058)
> * ? Description:
>   Ryerson Vision Lab Complex Document Information Processing dataset for document image classification

## Overview

RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) is a large-scale dataset for document image classification, introduced by Adam W. Harley, Alex Ufkes, and Konstantinos G. Derpanis at ICDAR 2015. The paper, titled "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," presented the dataset alongside an evaluation of deep CNN architectures for classifying document images.

The dataset is a subset of the IIT-CDIP Test Collection 1.0, which itself is derived from the Legacy Tobacco Document Library. It consists of 400,000 grayscale images evenly distributed across 16 document type classes. Images are sized so that their largest dimension does not exceed 1,000 pixels. The dataset has become a widely used benchmark for evaluating deep learning models in document analysis and classification tasks.

The 16 classes are: letter, form, email, handwritten, advertisement, scientific report, scientific publication, specification, file folder, news article, budget, invoice, presentation, questionnaire, resume, and memo.

## Statistics

- **Total images:** 400,000 grayscale images
- **Classes:** 16 (25,000 images per class)
- **Train:** 320,000 images
- **Validation:** 40,000 images
- **Test:** 40,000 images
- **Image format:** Grayscale TIF
- **Max dimension:** 1,000 pixels
- **Download size:** ~37 GB
- **Source:** IIT-CDIP Test Collection 1.0 / Legacy Tobacco Document Library

## Used in

[[OCR-free Document Understanding Transformer (Donut)]]
