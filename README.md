# 🌿 Forest Fire Severity Prediction – Machine Learning Project

## 📌 Overview

Forest fires are a major environmental concern, and predicting the severity of fires can help improve planning, prevention, and response.
This project uses machine learning algorithms to analyze real forest fire data and **predict the severity of a fire** based on environmental factors.

The project includes:

* Data loading and preprocessing
* Exploratory data analysis (EDA)
* Feature engineering
* Model training (Regression & Ensemble ML models)
* Performance evaluation
* Visualization of predictions

---

## 🧠 Problem Statement

Predict the **burn severity of a forest fire** using historical environmental parameters.

---

## 📂 Dataset Details

### Dataset Size

* **517 records**
* Used only for academic and experimental research purposes

### Dataset Attributes

The dataset contains the following input features:

| Attribute   | Description                                             |
| ----------- | ------------------------------------------------------- |
| X           | X-axis location coordinate                              |
| Y           | Y-axis location coordinate                              |
| Month       | Month of the year                                       |
| Day         | Day of the week                                         |
| FFMC        | Fine Fuel Moisture Code                                 |
| DMC         | Duff Moisture Code                                      |
| DC          | Drought Code                                            |
| ISI         | Initial Spread Index                                    |
| Temperature | Temperature in °C                                       |
| RH          | Relative Humidity                                       |
| Wind        | Wind speed in km/h                                      |
| Rain        | Rainfall in mm                                          |
| Area        | Burned area (used as the label for severity prediction) |

---

## 🧪 Machine Learning Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Ensemble Meta-models

Each model is trained, evaluated, and compared based on standard performance metrics.

---

## 📊 Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 🚀 System Workflow

1. Dataset loading
2. Data preprocessing & cleaning
3. Exploratory data analysis
4. Feature selection
5. Train-test split
6. Model training
7. Model evaluation
8. Visualization of results

---

## 💻 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Open Jupyter Notebook

```
jupyter notebook
```

### 4️⃣ Run the `.ipynb` file step-by-step

---

## 📈 Results

* Successfully trained ML models to predict forest fire severity.
* Compared performance of multiple ML algorithms.
* Visualized actual vs predicted results for better understanding.
