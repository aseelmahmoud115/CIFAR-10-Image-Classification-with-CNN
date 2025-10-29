# 🧠 CIFAR-10 Image Classification with CNN

This project implements a **Convolutional Neural Network (CNN)** using **Keras** and **TensorFlow** to classify images from the **CIFAR-10** dataset, which consists of 60,000 32x32 color images across 10 classes *(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)*.  
The goal is to build, train, and evaluate a deep learning model that can achieve high accuracy in identifying objects in the dataset.

---

## 🚀 Features
- Loads and preprocesses CIFAR-10 dataset.  
- Visualizes class distribution and sample images.  
- Builds a multi-layer CNN with Batch Normalization and Dropout.  
- Trains the model with accuracy/loss tracking and validation.  
- Displays accuracy/loss graphs for training progress.  
- Generates a confusion matrix and classification report.  
- Visualizes correct and misclassified examples.  
- Saves the trained model for future use.  

---

## 🛠️ Technologies Used
- Python  
- Google Colab  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- Scikit-learn *(Confusion Matrix & Evaluation)*  

---

## 🧪 How to Run
1. Open the `.ipynb` file in **Google Colab**.  
2. Run the cells in order to:
   - Load and preprocess the data.  
   - Build and train the CNN model.  
   - Evaluate the model's performance.  
   - Visualize predictions and misclassifications.  
   - Save the trained model locally.  

---

## 📊 Dataset
- **Name:** CIFAR-10  
- **Size:** 60,000 images *(50,000 train, 10,000 test)*  
- **Classes:** 10 object categories  
- **Source:** Built into `keras.datasets`  

---

## 📦 Output
- Trained CNN model saved as `.h5` file.  
- Accuracy and loss plots for training and validation.  
- Heatmap of confusion matrix.  
- Visual preview of predictions vs actual classes.  


