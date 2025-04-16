# Ubunifu (Creative)AI-JCIA-Hackathon 2025

# 🍑 Automatic Plum Sorting with EfficientNetB3 and CutMix

A computer vision system for **automated plum classification** using **transfer learning** and **advanced data augmentation techniques**. We used an augmentation technique like CutMix to enhance the model's ability to generalize by exposing it to a more diverse range of image variations, reducing overfitting and improving robustness against real-world inconsistencies. This model classifies plums into 6 quality categories based on visual features to support agricultural sorting and grading processes.

---

## 🧠 Project Overview

- **Goal**: Accurately classify images of plums into one of the six condition classes: `bruised`, `cracked`, `rotten`, `spotted`, `unaffected`, and `unripe`.
- **Approach**: Transfer learning using **EfficientNetB3**, enhanced with **CutMix** augmentation to improve model generalization.
- **Output**: A trained `.keras` model file, visualizations of learned feature space, and evaluation metrics.

---

## 📁 Directory Structure

```
plum-sorting/
│
├── data/
│   ├── bruised/
│   ├── cracked/
│   ├── rotten/
│   ├── spotted/
│   ├── unaffected/
│   └── unripe/
│
├── files/
│   ├── EfficientNetb3-final.keras      # Saved model
│   ├── log.csv                         # Training logs
│
├── plum-cutmix-effb3-6class-final.ipynb  # Main notebook
├── Plum_dataset_overview.ipynb # dataset overview
├── README.md
```

---

## I- Dataset Cleaning and Analysis (Plum_dataset_overview.ipynb)
This involves cleaning, analyzing, and preparing an image dataset of African plums for training machine learning models. The dataset initially contained class imbalances, possible duplicate and mislabelled images, and some quality issues.Below is a description of all operations performed:
from IPython.display import Markdown

## ✅ 1. Dataset Structure

- Directory: `african_plums_dataset/african_plums/`
- Each subfolder represents a class: `unaffected`, `cracked`, `bruised`, `rotten`, `spotted`, `unripe`.This makes it easy to load labeled data automatically, useful for training classification models using most deep learning frameworks 

---

## 📊 2. Initial Class Distribution

- Counted and visualized the number of images per class.
- Observed **significant class imbalance**: e.g., "unaffected" >> "cracked". This step reveals the need for techniques like class weighting, resampling, or augmentation.

---

## 🖼️ 3. Sample Visualization

- Displayed **3 random images per class**.
- Organized in rows with the class name as the row header.
- Helped identify visual diversity and outliers.

---

## 🧹 4. Cleaning the Dataset

### 🔁 a. Duplicate Detection
- Used **perceptual hashing (pHash)** with `imagehash` to detect near-identical images. Duplicate images artificially inflate model performance and introduce data leakage between train/test sets, harming real-world generalization.

### 👁️ b. Duplicate Review
- Displayed duplicate groups side by side.
- Automatically deleted all but one per group.

### 🕳️ c. Blank/Near-empty Image Detection
- This helps detect grayscale images with extremely low variation (e.g., all black, all white).
which can add noise, confuse the model, and waste training resources without adding value.

---

## 📉 5. Updated Class Distribution

- Recalculated image counts after cleaning.
- Replotted the bar chart to reflect the cleaned dataset.
- It helped to verify that our cleanup actions were applied properly, and whether further class balancing is needed (via upsampling, augmentation, etc.).

---

## 🧠 6. t-SNE Visualization

- Extracted feature vectors using **EfficientNetB0**.
- Applied **t-SNE** for dimensionality reduction.
- Plotted 2D clusters, color-coded by **actual class names**.
- Helps assess how separable classes are in feature space, whether images from different classes overlap (risk of misclassification) and whether visual clusters match expected labels.

---

## 🔗 7. Class Similarity Matrix

- Averaged embeddings per class.
- Computed **cosine similarity** between class centroids.
- Built a matrix to visualize **inter-class similarities** (e.g., overlap between spotted and bruised).
- This helps identify which classes are visually similar and likely to be confused and where model confusion might occur (e.g., "bruised" vs. "spotted").

---

## 💾 8. Save Cleaned Dataset

- Created new folder: `cleaned_african_plums_dataset/`
- Copied only cleaned, verified images, preserving the class structure.

---

## 🛠️ Libraries Used
- `matplotlib`, `Pillow`, `imagehash`
- `torch`, `torchvision`, `sklearn`, `numpy`
- `shutil`, `os`

## II- Main notebook (plum-cutmix-effb3-6class-final.ipynb)


### 1. 📥 Data Loading

- Data is loaded from a **cleaned folder structure**, where each subfolder corresponds to a class label.
- File paths and labels are collected, and labels are encoded using `LabelEncoder`.

### 2. 🧼 Preprocessing & Augmentation

- Images are resized to `(224, 224)` and normalized to `[0, 1]`.
- Augmentation:
  - Random flip (horizontal & vertical)
  - Random crop after resizing
- Labels are one-hot encoded using `tf.one_hot`.

### 3. 🧪 Dataset Splitting

- 80% of data → training & validation
- 20% of data → test set
- Within training/validation split: 90% training, 10% validation
- Created using `train_test_split()` from `sklearn`

### 4. 🔀 CutMix Augmentation

- Applied **on-the-fly** using `tf.data.map()`
- Random patches are extracted from different images and mixed
- Corresponding labels are **proportionally blended**

### 5. 🧠 Model Building

- `get_model()` dynamically loads a pretrained CNN from:
  - `EfficientNet`, `ResNet`, `VGG`, `MobileNet`, `Xception`, etc.
- Custom classification head:
  - Global pooling → Dense(512, ReLU) → Dropout(0.5) → Output (softmax for 6 classes)

```python
model = get_model('efficientb3', input_shape=(224, 224, 3), num_classes=6)
```

### 6. 🏋️ Training

- Loss: `categorical_crossentropy`
- Metrics: `accuracy`
- Optimizer: `Adam`
- Callbacks:
  - `ModelCheckpoint` (to save best weights)
  - `ReduceLROnPlateau`
  - `EarlyStopping` (optional)

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[...])
```

---

## 📊 Evaluation & Visualization

- Test accuracy and classification report computed after training
- **t-SNE** is used to visualize feature separability from the penultimate layer
- Custom batch visualization method is included to verify augmentations and label integrity

---

## 📦 Dependencies

To run our code, make sure to install the following packages:

Main dependencies include:

- `tensorflow`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `jupyter`
- `Pillow`

---

## 📈 Example Visualization

```python
# View a batch of CutMix-augmented images
show_batch(train_ds, class_names=label_encoder.classes_)

# Visualize learned feature space using t-SNE
visualize_features(model, test_ds)
```

---

## 📁 Output Files

- `EfficientNetb3-final.keras`: Saved trained model
- `log.csv`: Training and validation metrics over epochs

---

## 🧑‍💻 Author

**Ubunifu AI**  
- Djika Asmaou  Houma (Group leader)
- Tchouangwo Kamdem Sandrine Ariane
- Renaud Axel Eba
- Okpala Chibuike
- Dina Valdez Camille Chazeaud








**************************************************Thank you for reading**************************************************
