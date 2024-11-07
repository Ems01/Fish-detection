# Fish Detection using YOLOv8 on ROV images

## Project Statement in brief
This project presents a fish detection approach using the YOLOv8 model applied to images captured by Remotely Operated Vehicles (ROVs) in different ocean depths and marine environments. In addition to the standard detection pipeline, we developed an alternative weighted sum function that considers the best training epoch by evaluating multiple metrics. The methodology, challenges and results of the detection are discussed, demonstrating the effectiveness of the proposed technique in dealing with the variability of underwater images.

**Keywords**: fish detection, YOLOv8, ROV images, underwater detection, computer
vision

---

### 📗 Paper and slideshow download:

[Fish Detection using YOLOv8 on ROV images [PDF]](https://github.com/Ems01/Fish-detection/raw/main/Fish_Detection_using_Yolov8_on_ROV_images_Staffolani_Sardellini.pdf)

[Fish Detection using YOLOv8 on ROV images [PPTX]](https://github.com/Ems01/Fish-detection/raw/main/Fish_Detection_using_Yolov8_on_ROV_images_Staffolani_Sardellini.pptx)

---

<a name="index"></a>

## 📘 Table of Contents

* [🎯 Goal](#statement)
* [⚙️ Methodology](#methodology)
* [📈 Results](#results)
* [👨🏻‍💻 Authors](#Authors)

<a name="statement"/></a>

## 🎯 Goal

Daily fish abundance surveys in the Adriatic Sea are conducted using remotely operated vehicles (ROVs) near offshore structures.

Developing a tool to detect fish in these ROV images is critical to understanding the ecological impact of these structures. However, challenges such as species variability, seasonal environmental changes, and the quality of ROV-captured images complicate the task. Fish vary greatly in appearance, and environmental shifts affect visibility, making manual annotation challenging and consistent detection difficult in complex underwater conditions.

<a name="methodology"/></a>

## ⚙️ Methodology

### Cross-Validation and Dataset Aggregation

To improve generalizability, a 5-fold cross-validation approach was used. Smaller datasets were merged based on characteristics like brightness, depth, and water color, resulting in six datasets

An automated script divided each dataset into five portions for cross-validation. Each fold used one portion as the test set and the other four portions as the training set. Within each fold, 80% of the data was for training and 20% for testing, creating a balanced evaluation.

### Hyperparameters and Augmentation

Training was conducted with 5-fold cross-validation on a remote server with Docker and GPU support. The configurations used are summarized in Table below.

| **Model**       | **Epochs** | **Batch Size** | **Image Size** | **Augmentation** |
|-----------------|------------|----------------|----------------|-------------------|
| YOLOv8 Small    | 100        | 8              | 640x640       | No               |
| YOLOv8 Large    | 100        | 8              | 640x640       | No               |
| YOLOv8 Small    | 300        | 8              | 640x640       | No               |
| YOLOv8 Small    | 300        | 16             | 640x640       | Yes              |

*Table: Hyperparameters settings for each method.*

Augmentation settings used in the latter training include:
- **hsv_s=0.4**: 40% saturation variation
- **hsv_v=0.4**: 40% brightness variation
- **scale=0.5**: 50% scaling for size variation
- **fliplr=0.5**: 50% horizontal flipping

### Model Selection and Weighted Sum Function

Additionally, we implemented a custom weighted sum function to consider mAP, F1 score, and loss metrics, with the best epoch determined by balancing mAP, precision, recall, and minimizing box and class losses.

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
| Enrico Maria Sardellini | s1120355@studenti.univpm.it | [Ems01](https://github.com/Ems01) |
