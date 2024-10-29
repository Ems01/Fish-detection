# Fish Detection using ROV images

This repository contains the codes for the project for Computer Vision and Deep Learning exam on UNIVPM.

## Abstract
This project presents a fish detection approach using the YOLOv8 model applied to images captured by Remotely Operated Vehicles (ROVs) in different ocean depths and marine environments. In addition to the standard detection pipeline, we developed an alternative weighted sum function that considers the best training epoch by evaluating multiple metrics. The methodology, challenges and results of the detection are discussed, demonstrating the effectiveness of the proposed technique in dealing with the variability of underwater images.

**Keywords**: fish detection, YOLOv8, ROV images, underwater detection, computer
vision

---

### 📗 Thesis and slideshow download:

[Fish Detection using ROV images [PPTX]](todolink)

[Fish Detection using ROV images [PDF]](todolink)

---

<a name="index"></a>

## 📘 Table of Contents

* [🎯 Project Goal](#statement)
* [⚙️ Methodology](#methodology)
* [📈 Results](#results)
* [👨🏻‍💻 Authors](#Authors)

<a name="statement"/></a>

## 🎯 Project Goal

Daily fish abundance surveys in the Adriatic Sea are conducted using remotely operated vehicles (ROVs) near offshore structures.

Developing a tool to detect fish in these ROV images is critical to understanding the ecological impact of these structures. However, challenges such as species variability, seasonal environmental changes, and the quality of ROV-captured images complicate the task. Fish vary greatly in appearance, and environmental shifts affect visibility, making manual annotation challenging and consistent detection difficult in complex underwater conditions.

<a name="methodology"/></a>

## ⚙️ Methodology

## Methodology

### Dataset and Preprocessing

The initial dataset, provided by ENI, consisted of approximately 4,000 images organized into nine folders, each representing a different ROV source. These datasets included:
Agostino B, Amelia A, Barbara A, Barbara H, Cervia A, Cervia C, Emma, Fratello Cluster, and PCWB.

After an initial analysis, several data inconsistencies were identified:
- Numerous images lacked fish, resulting in no annotations.
- Labeling varied; some fish were labeled with biological names instead of a generic label.
- Many annotations were point-based, which did not provide bounding boxes suitable for object detection.

Among the datasets, *Emma* and *Barbara A* were of higher quality, but contained issues with large, unannotated fish shoals. This lack of consistency in annotation presented challenges for model training.

For preprocessing, we:
1. **Filtered Images**: Removed images without fish, resulting in a dataset of ~2,600 images.
2. **Unified Labels**: Replaced all fish labels with "fish" to simplify detection to a single class.
3. **Bounding Boxes**: Converted point-based annotations to centered 20x20 bounding boxes, acknowledging that this approximation may reduce accuracy.
4. **Adjusted Annotations**: Manually resized inappropriate bounding boxes through Roboflow.
5. **Removed Unannotated Shoals**: Excluded images containing dense, unannotated fish shoals, reserving them for future testing.

The processed datasets are summarized in Table 1 below.

| **Dataset**          | **Num of Images** | **Num of Labels** | **Labels per Image** |
|----------------------|-------------------|--------------------|-----------------------|
| Agostino B           | 37               | 390               | 10.5                 |
| Amelia A             | 20               | 61                | 3.1                  |
| Barbara A            | 1253             | 37,760            | 30.1                 |
| Barbara H            | 91               | 5,120             | 56.3                 |
| Cervia A             | 274              | 2,279             | 8.3                  |
| Cervia C             | 69               | 1,387             | 20.1                 |
| Emma                 | 768              | 21,744            | 28.3                 |
| Fratello Cluster     | 30               | 752               | 25.1                 |
| PCWB                 | 87               | 622               | 7.1                  |

### Cross-Validation and Dataset Aggregation

To improve generalizability, a 5-fold cross-validation approach was used. Smaller datasets were merged based on characteristics like brightness, depth, and water color, resulting in six datasets. These aggregations are outlined in Table 2 below.

| **Dataset**          | **Water Color** | **Depth** | **Avg Fish Size** |
|----------------------|-----------------|-----------|--------------------|
| Agostino B           | Dark green      | 5-10m     | 0.14%             |
| Amelia A             | Aqua green      | 6-8m      | 0.18%             |
| Barbara A            | Light blue      | 2-72m     | 0.13%             |
| Barbara H            | Light blue      | 5-40m     | 0.11%             |
| Cervia A             | Light blue      | 3-25m     | 0.10%             |
| Emma                 | Dark blue       | 0-70m     | 0.14%             |
| Fratello Cluster     | Light blue      | 2-10m     | 0.06%             |
| PCWB                 | Light green     | 4-6m      | 0.11%             |

An automated script divided each dataset into five portions for cross-validation. Each fold used one portion as the test set and the other four portions as the training set. Within each fold, 80% of the data was for training and 20% for testing, creating a balanced evaluation.

### Hyperparameters and Augmentation

Training was conducted with 5-fold cross-validation on a remote server with Docker and GPU support. The configurations used are summarized in Table 3 below.

| **Model**       | **Epochs** | **Batch Size** | **Image Size** | **Augmentation** |
|-----------------|------------|----------------|----------------|-------------------|
| YOLOv8 Small    | 100        | 8              | 640x640       | No               |
| YOLOv8 Large    | 100        | 8              | 640x640       | No               |
| YOLOv8 Small    | 300        | 8              | 640x640       | No               |
| YOLOv8 Small    | 300        | 16             | 640x640       | Yes              |

Augmentation settings used in the latter training include:
- **hsv_s=0.4**: 40% saturation variation
- **hsv_v=0.4**: 40% brightness variation
- **scale=0.5**: 50% scaling for size variation
- **fliplr=0.5**: 50% horizontal flipping

### Model Selection and Weighted Sum Function

For each fold, we selected the best epoch for inference using Ultralytics' `best.pt` file, which maximizes a fitness score prioritizing mAP across IoU thresholds. The fitness function is defined as:

\[
\text{fitness}_i = 0.0 \cdot P_i + 0.0 \cdot R_i + 0.1 \cdot \text{mAP}_{50,i} + 0.9 \cdot \text{mAP}_{50:95,i}
\]

Additionally, we implemented a custom weighted sum function to consider mAP, F1 score, and loss metrics, with the best epoch determined by:

\[
f_i = 0.25 \cdot \text{mAP}_{50,i} + 0.375 \cdot P_i + 0.375 \cdot R_i - 0.5 \cdot L_{\text{box},i} - 0.5 \cdot L_{\text{cls},i}
\]

This function maximizes inference performance by balancing mAP, precision, recall, and minimizing box and class losses.

<a name="results"/></a>

## 📈 Results

| **Method**             | **Precision** | **Recall** | **F1**  | **mAP50** | **CO2**   | **Time** |
|------------------------|--------------:|-----------:|--------:|----------:|-----------|----------|
| Large 100              | 0.616         | 0.592      | 0.601   | 0.558     | 2.92 kg   | 22 h     |
| Small 100              | 0.608         | 0.586      | 0.595   | 0.555     | 2.00 kg   | 18 h     |
| Small 300              | 0.630         | 0.598      | 0.611   | 0.571     | 4.85 kg   | 57 h     |
| Small 300 with Aug     | 0.639         | 0.599      | 0.616   | 0.574     | 3.90 kg   | 41 h     |

*Table: Average values of best model maximums for each fold for each method.*


<a name="Authors"/></a>

## 👨🏻‍💻 Authors

| Name              | Email                       | GitHub                                          |
|-------------------|-----------------------------|-------------------------------------------------|
| Federico Staffolani   | s1114954@studenti.univpm.it | [fedeStaffo](https://github.com/fedeStaffo)               |
| Enrico Maria Sardellini | s1120355@studenti.univpm.it | [Ems](https://github.com/Ems01) |