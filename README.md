# Ripe Strawberry Detection using YOLO and RT-DETR

This project demonstrates a complete workflow for training and comparing object detection models to identify ripe strawberries in images. The process begins with a baseline `YOLOv11` model and progresses to a more advanced `RT-DETR (Real-Time Detection Transformer)` model, enhanced with `SAHI` for superior accuracy on small objects.

## Table of Contents
- [Project Overview](#project-overview)
- [Why RT-DETR?](#why-RT-DETR?)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Data Splitting](#2-data-splitting)
  - [3. Model Training: From YOLO to RT-DETR](#3-model-training:from-yolo-to-rt-detr)
  - [4. Advanced Inference with SAHI](#4-advanced-inference-with-sahi)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Prediction and Export](#6-prediction-and-export)
- [Results](#results)

## Project Overview

The goal of this project is to build and train an efficient object detection model capable of locating ripe strawberries. This has applications in automated harvesting and yield estimation. I explore two architectures:

1. **YOLOv11:** A fast and popular CNN-based model that serves as our baseline.

2. **RT-DETR:** A state-of-the-art Transformer-based model that offers higher accuracy by better understanding the global context of an image.

## Why RT-DETR?
While YOLO models are known for their speed, we chose to advance to the `RT-DETR (Real-time DEtection TRansformer)` for this task due to several key advantages:

* **Superior Accuracy with Global Context:** Unlike traditional CNNs that process images through local receptive fields, RT-DETR's Transformer architecture allows it to view the image holistically. This "global context" helps it better understand complex scenes and the relationships between objects, which is crucial for distinguishing between overlapping strawberries and leaves.

* **State-of-the-Art Performance:** RT-DETR is a cutting-edge model that has been shown to outperform many real-time object detectors, including YOLO variants, on standard academic benchmarks in both speed and accuracy.

* **End-to-End Pipeline:** As a DETR-based model, it simplifies the detection process by removing the need for certain hand-designed components like Non-Maximum Suppression (NMS) during training, leading to a more streamlined, end-to-end pipeline.

These features make RT-DETR an excellent candidate for pushing the performance boundaries for our strawberry detection task.

## Dataset

The initial dataset is provided in a single XML file (`annotations.xml`) that contains annotations for multiple images. Each annotation includes:
- Image filename, width, and height.
- Bounding boxes for each strawberry with coordinates (`xtl`, `ytl`, `xbr`, `ybr`) and an occlusion status.

The notebook parses this XML file and processes it into a format suitable for training with YOLO.

## Methodology

The end-to-end process is implemented in the `yolov11.ipynb` Jupyter Notebook and can be broken down into the following key steps:

### 1. Data Preprocessing
The raw XML annotations are parsed and converted into a structured `pandas` DataFrame. The bounding box coordinates are then transformed from the top-left (`xtl`, `ytl`) and bottom-right (`xbr`, `ybr`) format to the YOLO format, which consists of:
- **`Xcent`**: The normalized x-coordinate of the center of the bounding box.
- **`Ycent`**: The normalized y-coordinate of the center of the bounding box.
- **`boxW`**: The normalized width of the bounding box.
- **`boxH`**: The normalized height of the bounding box.

The conversion formulas used are:
$$X_{cent} = \frac{x_{tl} + x_{br}}{2 \times image_{width}}$$
$$Y_{cent} = \frac{y_{tl} + y_{br}}{2 \times image_{height}}$$
$$boxW = \frac{x_{br} - x_{tl}}{image_{width}}$$
$$boxH = \frac{y_{br} - y_{tl}}{image_{height}}$$

A class label of `0` is assigned to all strawberry detections. The processed annotations are saved as individual `.txt` files for each image, with each file containing the label and the four normalized coordinates for every bounding box in that image.

### 2. Data Splitting
The dataset of images and their corresponding `.txt` annotation files are randomly shuffled and split into three sets:
- **Training set**: 25 images
- **Validation set**: 10 images
- **Test set**: The remaining images

These sets are organized into `train`, `valid`, and `test` directories, each containing `images` and `labels` subdirectories.

### 3. Model Training: From YOLO to RT-DETR
Two models are trained and fine-tuned on the custom strawberry dataset using the `ultralytics` library:
1. **Baseline Model:** A pretrained `YOLOv11n` model.
2. **Advanced Model:** A pretrained `RT-DETR` model, chosen for its advanced Transformer architecture. Both models are trained for 100 epochs with an image size of 640x640 on a GPU.

The training process is logged, showing metrics such as box loss, class loss, and mAP (mean Average Precision) for each epoch.

### 4. Advanced Inference with SAHI
To significantly improve detection accuracy on small strawberries, we apply SAHI (Slicing Aided Hyper Inference) during the prediction phase. This technique slices the input image into smaller, overlapping patches, runs the trained RT-DETR model on each patch, and then merges the results. This ensures even tiny objects are detected reliably.

### 5. Model Evaluation
After training, each model's performance is evaluated on the validation set. The key metrics are:
* **mAP50:** Mean Average Precision at an IoU threshold of 0.5.
* **mAP50-95:** Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.

### 6. Prediction and Export
The trained model is used to make predictions on a sample image from the test set to visually inspect its performance. Finally, the model is exported to the **ONNX** (Open Neural Network Exchange) format with dynamic input shapes, making it suitable for deployment across various platforms.

## Results
The models were trained for 100 epochs. The validation results for the best-performing models are compared below.

**YOLOv11n Results**

| Metric    | Value |
|-----------|-------|
| mAP50     | 0.968 |
| mAP50-95  | 0.779 |

**RT-DETR Results**

| Metric    | Value |
|-----------|-------|
| mAP50     | 0.981 |
| mAP50-95  | 0.844 |

## Conclusion
While the YOLOv11 model provides strong baseline performance, the RT-DETR model demonstrates superior accuracy, particularly in the more stringent mAP50-95 metric. This is attributed to its Transformer-based architecture. When combined with SAHI, the RT-DETR model becomes an exceptionally robust solution for detecting small objects in challenging, real-world scenarios.