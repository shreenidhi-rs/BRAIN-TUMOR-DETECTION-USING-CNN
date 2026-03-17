#  Brain Tumor Detection using Deep Learning (VGG19)

##  Overview

This project implements a **brain tumor classification system** using
MRI images and a **deep learning model based on VGG19**. The model
classifies images into:

-    **Non-tumorous**
-    **Tumorous**

The pipeline includes **data preprocessing, augmentation, model
training, evaluation, and prediction**.

------------------------------------------------------------------------

## Objectives

-   Detect brain tumors from MRI scans
-   Handle dataset imbalance using augmentation
-   Improve model accuracy using transfer learning
-   Evaluate performance with multiple metrics

------------------------------------------------------------------------

##  Dataset

The dataset consists of MRI images divided into:

    brain_tumor_dataset/
    ├── yes/   # Tumor images
    ├── no/    # Non-tumor images

------------------------------------------------------------------------

##  Workflow

### 1️ Data Preparation

-   Extract dataset from ZIP file
-   Rename images for consistency
-   Count and visualize class distribution

------------------------------------------------------------------------

### 2️ Data Augmentation

To reduce imbalance: - Rotation - Flipping (horizontal & vertical) -
Brightness adjustment - Shifting & shearing

Augmented data is stored in:

    augmented_data/
    ├── yes/
    ├── no/

------------------------------------------------------------------------

### 3 Preprocessing

Each image undergoes: - Grayscale conversion - Gaussian blur (noise
reduction) - Thresholding (Otsu method) - Erosion & dilation - Tumor
region cropping

------------------------------------------------------------------------

### 4️ Data Loading

-   Resize images to **240 × 240**
-   Normalize pixel values
-   Assign labels:
    -   `1 → Tumorous`
    -   `0 → Non-tumorous`

------------------------------------------------------------------------

### 5️ Data Splitting

Dataset is split into:

    tumorous_and_nontumorous/
    ├── train/
    ├── test/
    ├── valid/

------------------------------------------------------------------------

##  Model Architecture

###  Base Model

-   **VGG19 (Pre-trained on ImageNet)**
-   Frozen convolutional layers

###  Custom Layers

-   Flatten layer
-   Dense (4608 units, ReLU)
-   Dropout (0.2)
-   Dense (1152 units, ReLU)
-   Output layer (Softmax, 2 classes)

------------------------------------------------------------------------

##  Training Configuration

-   Optimizer: **SGD**
-   Loss: **Categorical Crossentropy**
-   Metrics: **Accuracy**

### Callbacks:

-   EarlyStopping
-   ModelCheckpoint
-   ReduceLROnPlateau

------------------------------------------------------------------------

##  Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix

------------------------------------------------------------------------

##  Results Visualization

-   Training vs Validation Accuracy
-   Training vs Validation Loss
-   Confusion Matrix heatmap

------------------------------------------------------------------------

##  Prediction

You can classify a new MRI image:

``` python
classify_image("image_path.jpg")
```

Output:

    Prediction: tumorous / nontumorous

------------------------------------------------------------------------

##  Requirements

Install dependencies using:

``` bash
pip install numpy pandas matplotlib seaborn opencv-python tensorflow scikit-learn imutils
```

------------------------------------------------------------------------

##  How to Run

1.  Upload dataset ZIP file\
2.  Run the notebook/script step by step\
3.  Train the model\
4.  Evaluate performance\
5.  Test with new images

------------------------------------------------------------------------

##  Key Highlights

-    Transfer Learning with VGG19\
-    Custom preprocessing (tumor cropping)\
-    Data augmentation for imbalance\
-    Comprehensive evaluation

------------------------------------------------------------------------

## Limitations

-   Requires large dataset for better generalization\
-   Cropping may fail on unclear images\
-   Model trained for binary classification only

------------------------------------------------------------------------

##  Future Improvements

-   Use more advanced architectures (ResNet, EfficientNet)
-   Add multi-class tumor classification
-   Deploy as a web application
-   Improve segmentation accuracy

------------------------------------------------------------------------

##  License

This project is for educational and research purposes.

------------------------------------------------------------------------

## Acknowledgements

-   TensorFlow & Keras
-   OpenCV
-   Scikit-learn

------------------------------------------------------------------------

##  Contact

For questions or contributions, feel free to reach out!
