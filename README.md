# 🚀 EDA Explorer – AI-Powered Data Analysis CLI

A lightweight command-line tool that performs **automated exploratory data analysis (EDA)** with intelligent insights, feature importance detection, and data quality checks.

Built to simulate how an **AI Data Analyst** works on real-world datasets.

## Demo

https://github.com/user-attachments/assets/7dff8329-71e8-4bca-ad01-404e75df8314


---

## ⚡ Key Highlights

- 🔍 **One-command analysis** → `analyze <dataset>`
- 🧠 **Auto target detection** for ML-based insights
- 📈 **Feature importance (no manual setup)**
- ⚠️ **Smart data warnings** (missing, ID columns, constants)
- 📊 **Correlation & outlier detection**
- 📁 **Auto report generation (.txt)**
- ⚡ **Handles large datasets efficiently (Parquet + sampling)**

---

## ⚡ Performance
-  Parquet storage
-  Large dataset handling (sampling)

## 🛠️ System Design
-  Command handler
-  Registry system
-  Modular agents (AnalysisAgent, etc.)
-  Logger integration


## 📦 Datasets
-  Titanic
-  Customer Churn
-  Credit Card Fraud

## 🚀 Future Enhancements
-  RAG-based EDA advisor
-  SQL query generator
-  Model training pipeline


## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Parquet (efficient storage)  

---

## ⚡ Quick Demo

```bash
load churn.csv
analyze churn








