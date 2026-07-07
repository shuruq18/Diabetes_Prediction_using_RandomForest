# 🩸 Diabetes Prediction using Random Forest

A machine learning project that predicts whether a patient has diabetes based on diagnostic health measurements, using a **Random Forest Classifier**.

> ⚠️ **Disclaimer:** This project is for **educational purposes only**. It is not a diagnostic tool and must not be used as a substitute for professional medical advice, testing, or diagnosis.

---

## 📌 Overview

Given a set of medical measurements (glucose level, blood pressure, BMI, age, etc.), the model predicts a binary outcome:
- **1** — Diabetic
- **0** — Not diabetic

---

## 📊 Dataset

The notebook reads `diabetes.csv` (the well-known **Pima Indians Diabetes Dataset**), where each row is a patient record with the following columns:

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Score indicating diabetes likelihood based on family history |
| `Age` | Age in years |
| `Outcome` | Target — 1 = diabetic, 0 = not diabetic |

---

## 🧠 Methodology

1. **Load data** — reads `diabetes.csv` and inspects it with `.info()` and `.head()`.
2. **Check for missing values** — `dataset.isnull().sum()`.
3. **Split features/target** — all columns except the last (`x`) vs. the `Outcome` column (`y`).
4. **Train/test split** — 80/20, `random_state=42`.
5. **Train model** — `RandomForestClassifier(n_estimators=90, criterion='gini', random_state=0)`.
6. **Evaluate**:
   - Confusion matrix (also visualized as a heatmap with `seaborn`)
   - Recall score
   - Full classification report (precision, recall, F1-score)
   - Overall accuracy
7. **Predict on new patients** — demonstrates predictions on two example patient records.

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `Diabetes_Prediction_using_RandomForest.ipynb` | Notebook: data loading, training, evaluation, and example predictions |
| `diabetes.csv` *(not included — supply your own)* | Pima Indians Diabetes dataset |

---

## ⚙️ Installation & Usage

### Requirements
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Run the notebook
1. Place `diabetes.csv` in the same working directory as the notebook (the notebook currently expects it at `/content/sample_data/`, a Google Colab default path — update this path if running locally).
2. Open `Diabetes_Prediction_using_RandomForest.ipynb` in Jupyter or Google Colab.
3. Run all cells to train the model and view the confusion matrix, recall, classification report, and accuracy.

### Predicting on a new patient
```python
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
classifier.predict(np.array([[6, 142, 72, 45, 0, 38.6, 0.627, 50]]))
```

---

## 🛠️ Tech Stack
- **Python**
- **pandas**, **numpy** – data handling
- **scikit-learn** – Random Forest, train/test split, evaluation metrics
- **seaborn**, **matplotlib** – confusion matrix visualization

---

## 📈 Future Improvements
- Handle missing/implausible values (e.g. `Glucose`, `BloodPressure`, `BMI` of 0 likely represent missing data, not true zeros) — `isnull().sum()` alone won't catch these
- Scale features (e.g. `StandardScaler`) — not strictly required for Random Forest, but useful if comparing against other models
- Add cross-validation and hyperparameter tuning (`GridSearchCV`) instead of fixed `n_estimators=90`
- Add ROC-AUC and precision-recall curves alongside the confusion matrix
- Persist the trained model with `joblib` for reuse in a script or simple web app
- Add a `requirements.txt`

---

## 👩‍💻 Author
**Shuruq Alghamdi**
GitHub: [@shuruq18](https://github.com/shuruq18)

---

## 📄 License
No license specified yet. Consider adding one (e.g., MIT) if you'd like others to reuse this work.
