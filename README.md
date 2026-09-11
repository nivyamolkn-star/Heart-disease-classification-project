# Heart Disease Classification using Machine Learning

##  Project Overview

This project demonstrates **Machine Learning classification** using four different algorithms to predict the presence of heart disease:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest

The models are trained and evaluated using a publicly available **Heart Disease dataset**. The project compares the accuracy of each model and visualizes the confusion matrix of the best-performing model.

> **Note:** The dataset used in this project was sourced from the publicly available **Zero to Mastery Machine Learning project by Mr. DBourke**. It is used here for educational and learning purposes.

##  Objectives

* Understand classification problems in Machine Learning
* Preprocess and standardize the dataset
* Train multiple classification algorithms
* Compare model accuracy
* Identify the best-performing model
* Generate and analyze a confusion matrix
* Evaluate the selected model using a classification report

##  Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

##  Machine Learning Models

### 1. K-Nearest Neighbors (KNN)

Classifies data points based on the classes of their nearest neighbors.

### 2. Support Vector Machine (SVM)

Finds an optimal decision boundary between different classes.

### 3. Decision Tree

Uses a tree-like structure of decisions to classify the data.

### 4. Random Forest

Combines multiple decision trees to improve prediction performance.

## Dataset

The project uses the Heart Disease dataset containing patient-related features and a `target` column representing the prediction outcome.

The dataset was obtained from:

**Zero to Mastery Machine Learning repository – Mr. DBourke**

Dataset URL:
https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv

The dataset is **not originally created by me** and is used for educational purposes.

##  Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Feature & Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Accuracy Comparison
   ↓
Best Model Selection
   ↓
Confusion Matrix
   ↓
Classification Report
```

## Model Evaluation

The accuracy of the following models is compared:

| Model         | Accuracy                    |
| ------------- | --------------------------- |
| KNN           | Calculated during execution |
| SVM           | Calculated during execution |
| Decision Tree | Calculated during execution |
| Random Forest | Calculated during execution |

The model with the highest accuracy is automatically selected as the **best model**.

##  Confusion Matrix

The project generates a confusion matrix for the best-performing model to visualize:

* True Positives
* True Negatives
* False Positives
* False Negatives

##  How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project

Open the project folder in **VS Code** or Jupyter Notebook.

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4. Run the Python file

```bash
python heart_disease_classification.py
```

The dataset will be loaded directly from the publicly available GitHub URL.

##  Learning Outcomes

Through this project, I learned how to:

* Work with real-world datasets
* Perform train-test splitting
* Apply feature scaling using `StandardScaler`
* Build classification models
* Compare different Machine Learning algorithms
* Evaluate models using accuracy
* Interpret confusion matrices
* Generate classification reports

##  Disclaimer

This project is created **for educational purposes only**. The predictions produced by these Machine Learning models should **not be used for medical diagnosis or clinical decision-making**.

##  Dataset Credit

Special credit to **Mr. DBourke / Zero to Mastery Machine Learning** for providing the publicly available dataset used in this project.

This project is an independent educational implementation and is **not an official Zero to Mastery project**.

##  Author

**Nivyamol K.N.**

This project was created as part of my learning journey in **Python, Machine Learning, and Data Science**.
