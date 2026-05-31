# 🩻 Pneumonia Detection using Deep Learning and Transfer Learning

## Overview

This project demonstrates an end-to-end deep learning pipeline for detecting pneumonia from chest X-ray images. The solution leverages medical imaging data in DICOM format, performs preprocessing and optimization through caching mechanisms, and compares multiple CNN-based architectures to identify the most effective model for pneumonia screening.

The project progresses from a custom CNN model built from scratch to a transfer learning approach using DenseNet121, highlighting the impact of advanced feature extraction techniques on medical image classification performance.

---

## Problem Statement

Pneumonia is a serious respiratory infection that can lead to severe complications if not detected early. Radiologists often review large volumes of chest X-rays, making automated screening systems valuable for prioritizing high-risk cases.

The objective of this project is to build a deep learning model capable of identifying lung opacity patterns associated with pneumonia from chest X-ray images while maximizing detection performance and minimizing missed cases.

---

## Dataset

* Dataset Type: Chest X-ray Images (DICOM)
* Classification Task: Binary Classification

  * Class 0: No Lung Opacity
  * Class 1: Lung Opacity / Pneumonia
* Dataset Size: ~30,000 images
* Image Format: DICOM (.dcm)

---

## Project Workflow

### 1. Exploratory Data Analysis

* Visualized random samples from each class
* Analyzed class imbalance
* Studied image quality variations and intensity distributions
* Generated key insights to guide preprocessing and modeling decisions

### 2. Data Preprocessing

* Converted DICOM images into NumPy arrays
* Implemented caching strategy for efficient data loading
* Resized images to:

  * 256×256
  * 224×224
* Normalized pixel values to 0–1 range
* Created train and validation datasets
* Built optimized generators for scalable training

### 3. Model Development

#### CNN Model (From Scratch)

* Convolutional Neural Network with:

  * Conv2D Layers
  * MaxPooling Layers
  * Dense Layers
  * Dropout Regularization

#### Transfer Learning Model

* DenseNet121 Pre-trained on ImageNet
* Added custom classification head
* Fine-tuned upper DenseNet layers using a low learning rate

### 4. Model Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix Analysis
* Threshold Optimization

---

## Results

| Model                    | Accuracy | Recall | Precision | F1 Score | ROC-AUC | PR-AUC |
| ------------------------ | -------- | ------ | --------- | -------- | ------- | ------ |
| CNN (256×256)            | 71.4%    | 75.0%  | 53.4%     | 0.624    | 0.779   | 0.578  |
| CNN (224×224)            | 72.4%    | 76.9%  | 54.4%     | 0.638    | 0.798   | 0.620  |
| DenseNet121 (Frozen)     | 78.6%    | 78.5%  | 62.9%     | 0.699    | 0.863   | 0.742  |
| DenseNet121 (Fine-Tuned) | 78.1%    | 80.8%  | 61.7%     | 0.700    | 0.863   | 0.741  |

---

## Key Findings

* Reducing image size from 256×256 to 224×224 improved model generalization and reduced computational overhead.
* Transfer learning significantly outperformed CNN models trained from scratch.
* DenseNet121 improved PR-AUC by over 20% compared to the baseline CNN.
* Fine-tuning improved recall and reduced false negatives, making the model more suitable for screening applications.
* Threshold optimization had a substantial impact on balancing recall and precision.

---

## Business Impact

This solution can be deployed as an AI-assisted screening tool to:

* Prioritize high-risk chest X-rays for radiologist review
* Reduce diagnostic turnaround time
* Improve early pneumonia detection
* Assist healthcare professionals in managing large imaging workloads

The model is intended to support clinical decision-making and should not be used as a standalone diagnostic system.

---

## Technologies Used

* Python
* TensorFlow / Keras
* DenseNet121
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit / Gradio
* Hugging Face Spaces

---

## Future Improvements

* Grad-CAM based explainability
* EfficientNet and ConvNeXt benchmarking
* Ensemble learning
* Focal Loss implementation
* Pneumonia localization and severity prediction
* Deployment with monitoring and retraining pipelines

---

## Conclusion

This project demonstrates how transfer learning can significantly improve medical image classification performance compared to traditional CNN architectures. DenseNet121 emerged as the strongest model, delivering high recall and robust discrimination capability, making it a suitable candidate for pneumonia screening applications.

The results further highlight that model architecture and feature extraction capabilities contribute more to performance improvements than simply increasing image resolution.
