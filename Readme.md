# ⚡ Energy Consumption Forecasting MLOps Pipeline

## 📌 Project Title

**Building a Reproducible ML Pipeline for Energy Consumption Forecasting**

---

## 🎯 Objective

The goal of this project is to design a **reproducible end-to-end Machine Learning pipeline** to predict household energy consumption using historical electricity usage data.

This project focuses not only on model building, but on **MLOps best practices**, including:

* Data versioning
* Pipeline automation
* Experiment tracking
* Configurable parameters
* Reproducible workflow

---

## 🏭 Domain

**Energy / Power Analytics**

Energy providers use such models to:

* Forecast demand
* Plan resource allocation
* Analyze usage trends
* Detect anomalies

---

## 📂 Dataset

**Household Power Consumption Dataset**

Contains minute-level power usage measurements of a house.

### Features

| Column                | Description                                      |
| --------------------- | ------------------------------------------------ |
| Global_active_power   | Total active power consumption (target variable) |
| Global_reactive_power | Reactive power                                   |
| Voltage               | Voltage in volts                                 |
| Global_intensity      | Current intensity                                |
| Sub_metering_1        | Kitchen appliances energy                        |
| Sub_metering_2        | Laundry room energy                              |
| Sub_metering_3        | Water heater & AC energy                         |
| Datetime              | Timestamp (engineered from Date + Time)          |

---

## 🧠 ML Problem

**Type:** Regression
**Target:** `Global_active_power`

---

## 🏗 Project Structure

```
energy-mlops-pipeline/
│
├── data/
│   ├── raw/                # Raw dataset (DVC tracked)
│   └── processed/          # Cleaned & feature engineered data
│
├── src/
│   ├── ingestion.py        # Load raw data
│   ├── preprocessing.py    # Clean missing values
│   ├── feature_engineering.py # Create time-based features
│   ├── train.py            # Model training + MLflow logging
│   └── evaluate.py         # Model evaluation
│
├── params.yaml             # Model & split parameters
├── dvc.yaml                # DVC pipeline definition
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline Workflow

```
Raw Data
   ↓
Data Ingestion
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Evaluation
```

---

## ⚙️ Technologies Used

| Tool             | Purpose                    |
| ---------------- | -------------------------- |
| **Python**       | Programming language       |
| **Scikit-Learn** | Machine learning model     |
| **DVC**          | Data & pipeline versioning |
| **MLflow**       | Experiment tracking        |
| **Pandas**       | Data processing            |

---

## 🤖 Model Used

**Random Forest Regressor**

Why?

* Handles non-linear relationships
* Works well with tabular data
* Robust to noise

---

## 🧪 Experiment Tracking (MLflow)

MLflow tracks:

* Hyperparameters (n_estimators, max_depth)
* RMSE
* R² score
* Saved model artifacts

To view experiments:

```bash
mlflow ui
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 🔁 Data Versioning & Pipeline Automation (DVC)

DVC ensures:

* Reproducible runs
* Automatic pipeline execution
* Version control for data & models

Run entire pipeline:

```bash
dvc repro
```

---

## ⚙️ Parameters Configuration

All parameters are stored in `params.yaml`:

```yaml
data:
  test_size: 0.2
  random_state: 42

model:
  n_estimators: 80
  max_depth: 12
```

Changing parameters → automatically triggers new experiment.

---

## 📈 Results

| Model Config                 | RMSE       | R²     |
| ---------------------------- | ---------- | ------ |
| RF (depth=12, estimators=80) | **~0.034** | ~0.999 |

Hyperparameter tuning showed deeper trees and more estimators reduced error.

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Initialize DVC

```bash
git init
dvc init
```

### 3️⃣ Run MLflow server

```bash
mlflow ui
```

### 4️⃣ Execute full pipeline

```bash
dvc repro
```

---

## 📌 Key MLOps Concepts Demonstrated

✔ Modular ML pipeline
✔ Data versioning with DVC
✔ Experiment tracking with MLflow
✔ Config-driven ML
✔ Reproducible workflow

---

## 🏁 Conclusion

This project demonstrates how to build a **production-style ML workflow**, integrating data engineering, model training, and experiment management in a reproducible manner.

---

## 👨‍💻 Author

**Rishabh Mishra**
MLOps Project — Energy Consumption Forecasting

---

Agar chahe to next mai:

* `requirements.txt`
* GitHub push commands
  bhi de du 🔥
