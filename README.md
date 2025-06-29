# Ripped Strawberry Detection using YOLOv8

This project demonstrates a complete workflow for training a YOLO (You Only Look Once) object detection model to identify ripe strawberries in images. The process covers data preprocessing from XML annotations, converting them to the YOLO format, training the model, and evaluating its performance.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Data Splitting](#2-data-splitting)
  - [3. Model Training](#3-model-training)
  - [4. Model Evaluation](#4-model-evaluation)
  - [5. Prediction and Export](#5-prediction-and-export)
- [Results](#results)
- [How to Use](#how-to-use)
  - [Prerequisites](#prerequisites)
  - [Directory Structure](#directory-structure)
  - [Running the Code](#running-the-code)

## Project Overview

The goal of this project is to build and train an efficient object detection model capable of locating ripe strawberries in images. This has applications in automated harvesting, yield estimation, and fruit quality assessment. We use the Ultralytics YOLOv8 framework for its state-of-the-art performance and ease of use.

## Dataset

The initial dataset is provided in a single XML file (`annotations.xml`) that contains annotations for multiple images. Each annotation includes:
- Image filename, width, and height.
- Bounding boxes for each strawberry with coordinates (`xtl`, `ytl`, `xbr`, `ybr`) and an occlusion status.

The notebook parses this XML file and processes it into a format suitable for training with YOLO.

## Methodology

The end-to-end process is implemented in the `object_detection.ipynb` Jupyter Notebook and can be broken down into the following key steps:

### 1. Data Preprocessing
The raw XML annotations are parsed and converted into a structured `pandas` DataFrame. The bounding box coordinates are then transformed from the top-left (`xtl`, `ytl`) and bottom-right (`xbr`, `ybr`) format to the YOLO format, which consists of:
- **`Xcent`**: The normalized x-coordinate of the center of the bounding box.
- **`Ycent`**: The normalized y-coordinate of the center of the bounding box.
- **`boxW`**: The normalized width of the bounding box.
- **`boxH`**: The normalized height of the bounding box.

The conversion formulas used are:
$$X_{cent} = \frac{x_{tl} + x_{br}}{2 \times \text{image\_width}}$$$$Y_{cent} = \frac{y_{tl} + y_{br}}{2 \times \text{image\_height}}$$$$\text{boxW} = \frac{x_{br} - x_{tl}}{\text{image\_width}}$$
$$\text{boxH} = \frac{y_{br} - y_{tl}}{\text{image\_height}}$$

A class label of `0` is assigned to all strawberry detections. The processed annotations are saved as individual `.txt` files for each image, with each file containing the label and the four normalized coordinates for every bounding box in that image.

### 2. Data Splitting
The dataset of images and their corresponding `.txt` annotation files are randomly shuffled and split into three sets:
- **Training set**: 25 images
- **Validation set**: 10 images
- **Test set**: The remaining images

These sets are organized into `train`, `valid`, and `test` directories, each containing `images` and `labels` subdirectories.

### 3. Model Training
A pretrained **YOLOv11n** model from the `ultralytics` library is used as the base. The model is then fine-tuned on the custom strawberry dataset. The training is configured with the following key parameters:
- **Epochs**: 100
- **Image Size**: 640x640
- **Device**: GPU ("0")

The training process is logged, showing metrics such as box loss, class loss, and mAP (mean Average Precision) for each epoch.

### 4. Model Evaluation
After training, the model's performance is evaluated on the validation set. The key metrics used for evaluation are:
- **mAP50**: Mean Average Precision at an IoU (Intersection over Union) threshold of 0.5.
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.

### 5. Prediction and Export
The trained model is used to make predictions on a sample image from the test set to visually inspect its performance. Finally, the model is exported to the **ONNX** (Open Neural Network Exchange) format with dynamic input shapes, making it suitable for deployment across various platforms.

## Results

The model was trained for 100 epochs, and the validation results for the best-performing model are as follows:

| Metric    | Value |
|-----------|-------|
| mAP50     | 0.968 |
| mAP50-95  | 0.779 |

These results indicate a high level of accuracy in detecting ripe strawberries in the validation dataset.