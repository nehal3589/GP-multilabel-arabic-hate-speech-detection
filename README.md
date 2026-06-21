# A Multi-Label System for Arabic Hate Speech Detection Incorporating Explainable AI

## Overview

This graduation project presents a multi-label framework for Arabic hate speech detection on social media content.

The proposed system simultaneously performs three classification tasks:

- Hate Speech Detection
- Dialect Identification
- Topic Identification

The framework combines textual and emoji-based features to capture both explicit and implicit forms of hate speech. In addition, Explainable Artificial Intelligence (XAI) techniques were incorporated to improve the interpretability and transparency of model predictions.

---

## Project Objectives

- Develop a multi-label classification framework for Arabic social media content.
- Detect hate speech, dialect type, and tweet topic simultaneously.
- Integrate textual and emoji-based features.
- Improve model transparency using Explainable AI (XAI).
- Support future research in dialect-aware Arabic hate speech detection.

---

## Methodology

The project was developed through four main phases:

1. Data Collection from X (Twitter)
2. Data Cleaning and Preprocessing
3. Machine Learning and Transformer Model Development
4. Model Evaluation and Explainability Analysis

---

## Dataset Development

- Arabic tweets were collected from X (Twitter).
- The dataset covers Saudi and Egyptian dialects.
- Political and religious topics were included.
- Manual annotation, validation, and preprocessing were applied.
- The final dataset consisted of **3,962 Arabic tweets**.

---

## Technologies Used

- Python
- Scikit-learn
- PyTorch
- Hugging Face Transformers
- Google Colab
- LIME (Explainable AI)

---

## Results

### Hate Speech Detection
- Best Models: **MARBERT** and **CAMeLBERT**
- Accuracy: **81%**

### Dialect Identification
- Best Model: **CAMeLBERT**
- Accuracy: **93%**

### Topic Identification
- Best Model: **MARBERT**
- Accuracy: **99%**

### Overall Performance
- CAMeLBERT achieved the best overall performance with **75% overall accuracy**.
- Transformer models outperformed traditional machine learning models across all tasks.
- Emoji-based features and XAI techniques improved prediction interpretability.

---

## Notice

Due to intellectual property and academic considerations, the source code, trained models, and full dataset are not publicly available.

This repository is intended to showcase the project, methodology, findings, and research contributions.

---

## Team Members

- [Nehal Alzahrani](https://github.com/nehal3589)
- [Joud Alhuthali](https://github.com/USERNAME2)
- [Reham Alsubhi](https://github.com/qrqll)
- [Layan Babkour](https://github.com/Layan-Babkour)

## Supervisor

Dr. Tahani Alqurashi

## Institution

Data Science Department  
College of Computing  
Umm Al-Qura University
