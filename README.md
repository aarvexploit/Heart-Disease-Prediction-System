# Heart Disease Prediction System

An end-to-end machine learning pipeline and interactive web application that predicts the presence of heart disease based on patient clinical parameters. The project compares **Logistic Regression** and **Decision Tree** classifiers and is deployed on Streamlit Cloud.

**Live Application:** [Heart Disease Prediction System on Streamlit Cloud](https://heart-disease-prediction-system-3kkfywr3xuappbw6u2sh9zd.streamlit.app/)

## Project Structure

```
heart-disease-prediction-system/
├── app.py                   # Streamlit web application for real-time predictions
├── requirements.txt         # Python dependencies required to run the project
├── README.md                # Project documentation
├── data/
│   ├── raw_data.csv         # Original UCI Heart Disease dataset (920 rows × 16 columns)
│   └── clean_data.csv       # Preprocessed and imputed dataset ready for modeling
├── models/
│   └── heart_models.pkl     # Serialized Logistic Regression, Decision Tree, scaler, and feature metadata
├── evaluations/
│   ├── confusion_matrices.png
│   ├── roc_curve.png
│   └── log_reg_coefficients.png
└── notebook/
    ├── data_evaluation.ipynb                  # Exploratory data analysis and missing value inspection
    ├── clean_data.ipynb        # Target creation, zero handling, and missing value imputation
    ├── encoding_data_and_model_training.ipynb       # Encoding, scaling, train-test split, and model training
    └── model_evaluation.ipynb     # Performance comparison, ROC curves, and feature importance
```

## Dataset & Preprocessing

* **Source Dataset (`data/raw_data.csv`):** Contains 920 patient records across 16 clinical attributes.
* **Target Variable:** Binarized `num` into `target` (`0` = No Heart Disease, `1` = Heart Disease Present).
* **Feature Cleaning (`data/clean_data.csv`):**
  * Dropped non-predictive identifiers and high-missingness columns (`id`, `dataset`, `num`, `ca`, `thal`).
  * Replaced biologically invalid `0` values in `chol` (serum cholesterol) and `trestbps` (resting blood pressure) with `NaN`.
  * Imputed numeric features (`trestbps`, `chol`, `thalch`, `oldpeak`) using column medians.
  * Imputed categorical/boolean features (`fbs`, `restecg`, `exang`) using column modes, and filled missing `slope` entries with `'unknown'`.
  * Applied binary mapping for `sex` and one-hot encoding (`drop_first=True`) for `cp`, `restecg`, and `slope`.

## Model Training & Evaluation

The dataset was split using an **80/20 stratified train-test split** (`random_state=42`). Features were standardized with `StandardScaler` (fit strictly on the training set) for Logistic Regression, while the Decision Tree (`max_depth=4`) was trained on unscaled features.

| Model | Test Accuracy | Precision | Recall | ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82.61%** | **83.02%** | **86.27%** | **0.8886** |
| **Decision Tree** (`max_depth=4`) | 78.26% | 79.81% | 81.37% | 0.8588 |

* **Top Predictive Features:** Exercise-induced angina (`exang`), chest pain type (`cp`), sex, age, maximum heart rate achieved (`thalch`), and ST depression (`oldpeak`).

### Evaluation Visualizations

#### Confusion Matrices
![Confusion Matrices](evaluations/confusion_matrices.png)

#### ROC Curve Comparison
![ROC Curve](evaluations/roc_curve.png)

#### Logistic Regression Feature Effects
![Logistic Regression Coefficients](evaluations/log_reg_coefficients.png)

## Local Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aarvexploit/Heart-Disease-Prediction-System.git
   cd Heart-Disease-Prediction-System
   ```

2. **Create and activate a virtual environment (optional):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Web Application Features

* **Interactive Clinical Inputs:** Enter patient metrics including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, and ST slope.
* **Side-by-Side Model Comparison:** View predicted heart disease probabilities and diagnostic classifications simultaneously from both Logistic Regression and Decision Tree models.
* **Feature Importance Sidebar:** Displays the top clinical drivers influencing Decision Tree predictions alongside an expandable view of the encoded model input vector.
