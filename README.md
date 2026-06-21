# A Multi-Label System for Arabic Hate Speech Detection Incorporating Explainable AI

## Overview

This repository presents a research-based graduation project that investigates Arabic hate speech detection using a multi-label classification framework for social media content.

The proposed framework simultaneously performs three classification tasks:

* Hate Speech Detection
* Dialect Identification
* Topic Identification

The framework combines textual and emoji-based features to capture both explicit and implicit forms of hate speech. In addition, Explainable Artificial Intelligence (XAI) techniques were incorporated to improve the interpretability and transparency of model predictions.

This work is intended for academic and research purposes and focuses on experimentation, model development, and evaluation rather than developing a production application or web-based system.

---

## Project Objectives

* Develop a multi-label classification framework for Arabic social media content.
* Detect hate speech, dialect type, and tweet topic simultaneously.
* Integrate textual and emoji-based features.
* Improve model transparency using Explainable AI (XAI).
* Build a multi-label Arabic dataset covering Saudi and Egyptian dialects across political and religious contexts.
* Support future research in dialect-aware and interpretable Arabic hate speech detection systems.

---

## Methodology

The project was developed through four main phases:

1. Data Collection from X (Twitter)
2. Data Cleaning and Preprocessing
3. Machine Learning and Transformer Model Development
4. Model Evaluation and Explainability Analysis

---

## Dataset Development

* Arabic tweets were collected from X (Twitter).
* The dataset covers Saudi and Egyptian dialects.
* Political and religious topics were included.
* Manual annotation, validation, and preprocessing were applied.
* The final dataset consisted of **3,962 Arabic tweets**.

---

## Technologies Used

* Python
* Scikit-learn
* PyTorch
* Hugging Face Transformers
* Google Colab
* LIME (Explainable AI)

---

## Results

### Hate Speech Detection

* Best Models: **MARBERT** and **CAMeLBERT**
* Accuracy: **81%**

### Dialect Identification

* Best Model: **CAMeLBERT**
* Accuracy: **93%**

### Topic Identification

* Best Model: **MARBERT**
* Accuracy: **99%**

### Overall Performance

* CAMeLBERT achieved the best overall performance with **75% overall accuracy**.
* Transformer models outperformed traditional machine learning models across all tasks.
* Emoji-based features and XAI techniques improved prediction interpretability and model transparency.

---

## Repository Structure

This repository contains:

### Traditional Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes
* XGBoost

### Transformer-Based Models

* AraBERT
* CAMeLBERT
* MARBERT
* XLM-RoBERTa

### Additional Resources

* Explainable AI (XAI) implementation using LIME.
* Project poster.
---

## Dataset Availability

The dataset used in this study is not publicly available due to privacy, academic, and intellectual property considerations.

---

## Team Members

* [Nehal Alzahrani](https://github.com/nehal3589)
* [Joud Alhuthali](https://github.com/USERNAME2)
* [Reham Alsubhi](https://github.com/qrqll)
* [Layan Babkour](https://github.com/Layan-Babkour)

## Supervisor

Dr. Tahani Alqurashi

## Institution

Data Science Department
College of Computing
Umm Al-Qura University

