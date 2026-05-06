# SignLearn Data Repository

## Overview
This directory contains the datasets used for training and validating the SignLearn sign language recognition system. It includes both static images for alphabet/digit classification and dynamic sequence placeholders.

## Structure
- `raw/`: Unprocessed datasets exactly as downloaded from sources.
- `processed/`: Cleaned, resized, and normalized data ready for model training.
- `external/`: Third-party data or metadata used for augmentation or reference.

## Datasets

### 1. ASL Alphabet
- **Source**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) / Mendeley Data
- **Classes**: 29 (A-Z, space, del, nothing)
- **Format**: 200x200 RGB images
- **Count**: ~3,000 images per class
- **Status**: Static images only.

### 2. ASL Digits
- **Source**: [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)
- **Classes**: 10 (0-9)
- **Format**: 100x100 RGB images
- **Status**: Static images only.

## Known Issues
- **Background Consistency**: Many images have high contrast backgrounds which may cause the model to overfit to specific environments.
- **Static vs. Dynamic**: These datasets are **static**. They do not provide temporal motion data, which is required for the "Dynamic Words" category in our vocabulary.
- **Lighting**: Variation in lighting across sets might require heavy data augmentation.
- **Hand Orientation**: Primarily single-angle shots; lacks 3D variation.
