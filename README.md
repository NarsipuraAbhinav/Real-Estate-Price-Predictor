# 🏠 Real Estate Price Predictor

A machine learning project that predicts housing prices using the classic **Boston Housing Dataset**. Built with Python and scikit-learn, the model is trained on 13 socio-economic and structural features to estimate the median value of owner-occupied homes.

---

## 📌 Overview

This project walks through the end-to-end machine learning pipeline — from data exploration and preprocessing to model training and inference. The trained model is serialized using `joblib` for easy reuse without retraining.

---

## 📁 Repository Structure

```
Real-Estate-Price-Predictor/
│
├── Real Estate.ipynb        # Main notebook: EDA, preprocessing & model training
├── Model Usage.ipynb        # Notebook demonstrating how to load and use the saved model
├── Untitled.ipynb           # Experimental / scratch notebook
│
├── RealEstate.joblib        # Serialized trained model
│
├── housing.data             # Raw Boston Housing Dataset
├── housing.names            # Dataset attribute descriptions
├── data.csv                 # Processed/alternate dataset
└── loan-predictionUC.csv    # Additional dataset (loan prediction)
```

---

## 📊 Dataset

The project uses the **Boston Housing Dataset** — a classic regression benchmark from the StatLib library (Carnegie Mellon University).

| Property | Details |
|---|---|
| **Instances** | 506 |
| **Features** | 13 continuous + 1 binary |
| **Target** | `MEDV` — Median home value (in $1000s) |
| **Missing Values** | None |

### Features

| # | Feature | Description |
|---|---|---|
| 1 | `CRIM` | Per capita crime rate by town |
| 2 | `ZN` | Proportion of residential land zoned for lots >25,000 sq.ft. |
| 3 | `INDUS` | Proportion of non-retail business acres per town |
| 4 | `CHAS` | Charles River dummy variable (1 if tract bounds river, else 0) |
| 5 | `NOX` | Nitric oxides concentration (parts per 10 million) |
| 6 | `RM` | Average number of rooms per dwelling |
| 7 | `AGE` | Proportion of owner-occupied units built prior to 1940 |
| 8 | `DIS` | Weighted distances to five Boston employment centres |
| 9 | `RAD` | Index of accessibility to radial highways |
| 10 | `TAX` | Full-value property-tax rate per $10,000 |
| 11 | `PTRATIO` | Pupil-teacher ratio by town |
| 12 | `B` | 1000(Bk - 0.63)² where Bk is the proportion of Black residents by town |
| 13 | `LSTAT` | % lower status of the population |
| 14 | `MEDV` | **Target** — Median value of owner-occupied homes in $1000s |

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib jupyter
```

### Clone the Repository

```bash
git clone https://github.com/NarsipuraAbhinav/Real-Estate-Price-Predictor.git
cd Real-Estate-Price-Predictor
```

### Run the Notebooks

```bash
jupyter notebook
```

Open `Real Estate.ipynb` to explore the full pipeline, or `Model Usage.ipynb` to see how to load and use the pre-trained model.

---

## 🔮 Using the Pre-trained Model

The saved model (`RealEstate.joblib`) can be loaded directly for inference:

```python
import joblib
import numpy as np

# Load the model
model = joblib.load('RealEstate.joblib')

# Example input: [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]
sample = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]])

predicted_price = model.predict(sample)
print(f"Predicted Home Value: ${predicted_price[0] * 1000:,.2f}")
```

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Notebooks:** Jupyter
- **ML Library:** scikit-learn
- **Data Handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Serialization:** joblib

---
