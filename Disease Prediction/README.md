# 🧠 Disease Prediction Using Machine Learning

A Machine Learning-based system to predict diseases from symptoms provided by the user. This project demonstrates how machine learning can be applied to healthcare by predicting potential diseases using a dataset containing symptoms and diagnoses.

---

## 🚀 Features

- Predict diseases based on symptoms.
- Uses multiple ML algorithms (SVC, Random Forest, Naive Bayes).
- Evaluates and compares model performance using a confusion matrix.
- Simple CLI for entering symptoms.
- Combines model predictions for robust results.

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🗃️ Dataset

The dataset consists of:
- **132 columns**: Represent different symptoms.
- **1 column**: Represents the target label – `prognosis` (disease name).

Two CSV files are provided:
- `Training.csv` for model training.
- `Testing.csv` for evaluation.

---

## 🧼 Data Preparation

Before training, the data needs to be cleaned and encoded:

- All symptom columns are already binary (0 or 1).
- The target column `prognosis` (disease name) is label-encoded into numerical values so that machine learning models can process it.
- No missing values, so preprocessing is minimal.

---

## 🛠️ Model Building

We train three different classifiers on the dataset:

- **Support Vector Classifier (SVC)**
- **Naive Bayes Classifier**
- **Random Forest Classifier**

After training, we evaluate each model and later combine their predictions to improve the accuracy and robustness of the final output.

---



## 📉 Model Evaluation:

To evaluate our machine learning models, we use a **confusion matrix**—a powerful tool for understanding how well the model is performing beyond just accuracy.
A **confusion matrix** is a table used to describe the performance of a classification model.



### 📊 Key Components:

For **binary classification** (e.g., predicting disease or not):

- **True Positive (TP)**: Model correctly predicts the positive class.
- **True Negative (TN)**: Model correctly predicts the negative class.
- **False Positive (FP)**: Model incorrectly predicts the positive class.
- **False Negative (FN)**: Model incorrectly predicts the negative class.

For **multi-class classification** (like ours):

- Rows represent **actual classes (true diseases)**.
- Columns represent **predicted classes (model’s output)**.
- Each cell `[i][j]` tells how often class `i` was predicted as class `j`.
- The **diagonal** shows correct predictions.
- **Off-diagonal** values indicate misclassifications.


### ✅ Why Use It?

- Helps visualize **which diseases are being confused** by the model.
- Gives insight into **model bias** toward certain diseases.
- More informative than accuracy alone (especially with class imbalance).
- Supports the calculation of:
  - **Precision**: TP / (TP + FP)
  - **Recall**: TP / (TP + FN)
  - **F1-Score**: Harmonic mean of precision and recall

---

## 🔍 Final Prediction Logic

Once all models are trained:

- We collect predictions from each classifier.
- Use **majority voting** to select the final disease.
- This ensemble approach ensures that even if one model misclassifies, the final result is more likely to be accurate.

---

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/disease-prediction-ml.git
cd disease-prediction-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python disease_prediction.py
```

---

## ✅ Example Input

```text
Enter symptoms (comma-separated): itching, skin_rash, nodal_skin_eruptions
```

---

## 📊 Sample Accuracy (varies per run)

- SVC: ~97%
- Random Forest: ~98%
- Naive Bayes: ~90%

---
