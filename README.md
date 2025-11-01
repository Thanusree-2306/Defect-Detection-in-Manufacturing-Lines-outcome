# Defect Detection in Manufacturing Lines

**Defect Detection in Manufacturing Lines** is a deep learning–based project designed to automatically identify defective products in real-time manufacturing processes.  
By leveraging advanced computer vision and neural network models, this system helps improve production efficiency, reduce manual inspection time, and ensure consistent product quality.

---

## 📌 Project Overview

This project uses **deep learning** and **image processing** techniques to classify manufactured items as **defective** or **non-defective**.  
It can be integrated into smart factories or industrial automation systems for **real-time quality control**.

Key tasks include:
- Image preprocessing and data augmentation  
- Model training using **Convolutional Neural Networks (CNNs)**  
- Evaluation of model accuracy, precision, recall, and F1-score  
- Visualization of training metrics and prediction outcomes  

---

## 📂 Project Structure
├── Defect_Detection.ipynb # Main Jupyter/Colab Notebook
├── data/ # Dataset (train/test images)
│ ├── defective/
│ └── non_defective/
├── models/ # Saved trained models
│ ├── cnn_model.keras
│ └── model_history.json
├── results/ # Plots and test predictions
│ ├── confusion_matrix.png
│ ├── accuracy_curve.png
│ └── sample_predictions.png
└── README.md


---

## ⚙️ Features

- 🧹 **Automated image preprocessing** (resizing, normalization, augmentation)  
- 🧠 **CNN-based defect classification**  
- 📈 **Training visualization** (loss and accuracy curves)  
- 🧪 **Evaluation metrics:** Accuracy, Precision, Recall, F1-Score  
- 🧩 **Confusion matrix and result visualization**  
- 💾 **Model saving and reusability**

---

## 📊 Dataset

You can use any manufacturing defect dataset containing images of **defective** and **non-defective** products.  
For example:
- [Kaggle: Surface Crack Detection Dataset](https://www.kaggle.com/datasets/arunrk7/surface-crack-detection)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

The notebook assumes the following folder structure:
data/
├── defective/
└── non_defective/


---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Defect-Detection-in-Manufacturing-Lines.git
   cd Defect-Detection-in-Manufacturing-Lines


Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the notebook (locally or in Google Colab):

bash
Copy code
jupyter notebook Defect_Detection.ipynb
View results and saved models inside /results and /models.

✅ Results
Performance Metrics:

Accuracy: ~98%

Precision, Recall, F1-Score

Confusion Matrix Visualization

Tech Stack

Python 3

TensorFlow / Keras

NumPy, Pandas, Matplotlib, OpenCV

scikit-learn
