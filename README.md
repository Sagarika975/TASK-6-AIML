K-Nearest Neighbors (KNN) Classification – Iris Dataset

A simple yet complete machine learning project demonstrating KNN Classification using the classic Iris flower dataset.

📂 Project Structure
📁 KNN-Classification
│── KNN_Iris_Notebook.ipynb
│── Iris.csv
│── README.md

🧠 Project Overview

This project implements the K-Nearest Neighbors (KNN) algorithm to classify iris flower species based on their measured features.

🔍 Key Steps

Load & explore the dataset

Data preprocessing (scaling, feature selection)

Train-test split

Test multiple K values

Select best K

Evaluate using accuracy & confusion matrix

Plot K vs Accuracy

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

📊 Dataset

The project uses the Iris Dataset, which contains:

150 samples

4 numerical features

3 flower species

Target: Species classification

Dataset File: Iris.csv

🚀 How to Run
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn matplotlib

2️⃣ Run the Notebook

Open the notebook:

jupyter notebook KNN_Iris_Notebook.ipynb

🧪 Model Training Code (Summary)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
pred = model.predict(X_test)

📈 Results
✔ K vs Accuracy Graph

Shows how accuracy changes for K = 1–10.

✔ Best K

best_k = 3 (also 5–10 achieve 100%)

✔ Confusion Matrix

Perfect classification:

[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]

📚 What You’ll Learn

Instance-based learning

Euclidean distance

Choosing optimal K

How KNN works

Feature scaling importance

📝 Future Enhancements

Add decision boundary visualization

Add hyperparameter tuning using GridSearchCV

Deploy model using Flask/Streamlit

Convert project into a full ML pipeline

