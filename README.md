# Ubunifu (Creative) AI - JCIA Hackathon 2025

## 🍑 Automatic Plum Sorting with EfficientNetB3 and CutMix

A computer vision system for **automated plum classification** using **transfer learning** and **advanced data augmentation techniques**. We used an augmentation technique like CutMix to enhance the model's ability to generalize by exposing it to a more diverse range of image variations, reducing overfitting and improving robustness against real-world inconsistencies. This model classifies plums into 6 quality categories based on visual features to support agricultural sorting and grading processes.

---

## 🧠 Project Overview

- **Goal**: Accurately classify images of plums into one of the six condition classes: `bruised`, `cracked`, `rotten`, `spotted`, `unaffected`, and `unripe`.
- **Approach**: Transfer learning using **EfficientNetB3**, enhanced with **CutMix** augmentation to improve model generalization.
- **Output**: A trained `.keras` model file, visualizations of learned feature space, and evaluation metrics.

---

## 📁 Directory Structure

```

├── cleaned_african_plums_dataset/
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
├── Plum_cutmix_effb3_6class_Ubunifu_AI.ipynb  # Main notebook
├── Plum_dataset_overview_Ubunifu_AI.ipynb # dataset overview
├── Streamlit_App_plums_detect_Ubunifu_AI.py # Web application 
├── README.md
```

## 📘 1. `Plum_dataset_overview_Ubunifu_AI.ipynb` — *Dataset Cleaning & Exploration*

This notebook performs a detailed **analysis and preprocessing of the plum image dataset**. It includes:

- 📊 **Class distribution analysis**  
- 🧹 **Duplicate image detection and removal**  
- 🏷️ **Mislabelled sample identification**  
- 🕳️ **Detection of empty or near-empty images**  
- 📉 **Updated class distribution after cleaning**
- **Saved cleaned dataset in cleaned_african_plums_dataset**

> 🔍 **Goal**: Improve data quality before training by identifying and cleaning problematic samples.


## 📗 2. `Plum_cutmix_effb3_6class_Ubunifu_AI.ipynb` — *Model Training with CutMix and EfficientNetB3*

This notebook builds and trains a **Convolutional Neural Network** using **EfficientNetB3** and advanced augmentation techniques such as **CutMix**. It includes:

- 📦 Data loading and augmentation pipeline using `tf.data`
- 🔁 Data splitting into training, validation, and test sets
- ⚖️ Handling of class imbalance using **cutmix augmentation**
- 🧠 Model architecture based on **EfficientNetB3**
- 📈 Metrics & performance visualizations (accuracy, F1-score, t-SNE, confusion matrix)
- 💾 Saves the trained model and logs

> 🚀 **Training Result**: The model achieves ~72% top-1 accuracy and ~99.5% top-k accuracy, showing strong performance in classifying "unaffected" and "unripe" categories.


## ▶️ How to Run These Notebooks

### Step 1: Clone this repository
```bash
git clone https://github.com/your-username/plum-classification-ai.git
cd plum-classification-ai
```

### Step 2: Set up your environment
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python
```

### Step 3: Prepare your dataset

Your dataset folder should be organized as follows:

```
cleaned_african_plums_dataset/
├── bruised/
├── cracked/
├── rotten/
├── spotted/
├── unaffected/
└── unripe/
```

You can upload the raw dataset from Kaggle (https://www.kaggle.com/datasets/arnaudfadja/african-plums-quality-and-defect-assessment-data) and run the **dataset overview notebook** to clean it and save it to cleaned_african_plums_dataset


### Step 4: Run the notebooks

Launch Jupyter Notebook or Colab:

```bash
jupyter notebook Plum_dataset_overview_Ubunifu_AI.ipynb
jupyter notebook Plum_cutmix_effb3_6class_Ubunifu_AI.ipynb
```

> ✅ *Recommended: Use Google Colab for faster training (especially with GPU like A100).*


## 🌐 How to Run the Streamlit App

Once the model is trained and saved (e.g., `EfficientNetb3-final.keras`), you can deploy it using the **Streamlit app** to make predictions on new plum images or videos.

### 🔧 1. Install Streamlit and required packages
Make sure the following packages are installed:

```bash
pip install streamlit tensorflow opencv-python pillow numpy pandas plotly scikit-learn matplotlib
```

### 📁 2. Project Structure Example

Ensure your project directory looks like this:

```
plum-classification-ai/
├── files/
│   └── EfficientNetb3-final.keras           # trained model
├── app.py                                   # Streamlit app script
├── image_prediction_result.csv              # (auto-generated)
├── video_prediction_results.csv             # (auto-generated)
├── t-SNE.png                                # optional for visualization
├── metrics.png                              # optional for metrics

```

### ▶️ 3. Launch the app locally

In your terminal or Anaconda prompt:

```bash
streamlit run app.py
```

This will open a browser window with the interactive plum classification interface.


### 📦 4. Features of the App

- 📷 Upload an image or video of plums
- 🧠 The model will predict one of 6 classes:
  `bruised`, `cracked`, `rotten`, `spotted`, `unaffected`, `unripe`
- 📈 View class probabilities, confidence scores, and frame-by-frame predictions
- 💾 CSV files with results are saved automatically
- 📊 Optional t-SNE and metrics plots are displayed if the files are available
  
## 🧑‍💻 Authors
**Ubunifu AI**  
- Djika Asmaou  Houma (Group leader)
- Tchouangwo Kamdem Sandrine Ariane
- Renaud Axel Eba
- Okpala Chibuike
- Dina Valdez Camille Chazeaud





**************************************************Thank you for reading**************************************************



