# A Multi-Label System for Arabic Hate Speech Detection Incorporating Explainable AI

## Overview

This repository presents a research-based graduation project that addresses the challenge of Arabic hate speech detection in social media content through a multi-label classification framework.

Unlike traditional approaches that focus solely on detecting hate speech, the proposed framework simultaneously performs three related tasks:

* Hate Speech Detection
* Dialect Identification
* Topic Identification

By combining these tasks within a single framework, the system provides a richer understanding of Arabic social media content and captures contextual information that can improve hate speech analysis.

To support this research, a custom multi-label Arabic dataset was developed from scratch by collecting tweets from X (Twitter) using carefully selected keywords. The collected tweets were manually reviewed, annotated, validated, and preprocessed before being used for model training and evaluation.

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

A custom Arabic dataset was created specifically for this study.

* Tweets were collected from X (Twitter) using predefined keywords related to political and religious contexts.
* The dataset includes Saudi and Egyptian dialects.
* Each tweet was manually annotated with three labels: hate speech, dialect type, and topic.
* Data validation, cleaning, and preprocessing were performed to ensure quality and consistency.
* The final dataset consisted of **3,962 Arabic tweets**.

The resulting dataset enabled the development and evaluation of a multi-label classification framework capable of performing all three tasks simultaneously.

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
