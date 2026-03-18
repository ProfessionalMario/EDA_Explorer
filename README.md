# 🚀 EDA Explorer – AI-Powered Data Analysis CLI

A lightweight CLI tool that automates exploratory data analysis (EDA) with intelligent insights, feature importance detection, and data quality checks.

Designed to simulate how an **AI Data Analyst** works on real-world datasets.

---

## ⚡ Key Highlights

- 🔍 One-command analysis → `analyze <dataset>`
- 🧠 Auto target detection for ML-based insights
- 📈 Feature importance (no manual setup)
- ⚠️ Smart data warnings (missing, ID columns, constants)
- 📊 Correlation & outlier detection
- 📁 Auto report generation (.txt)
- ⚡ Efficient handling of large datasets (Parquet + sampling)

---

## 🎬 Demo

👉 Full demo: https://github.com/user-attachments/assets/7dff8329-71e8-4bca-ad01-404e75df8314

https://github.com/user-attachments/assets/7dff8329-71e8-4bca-ad01-404e75df8314

---

## 📊 Example Output
Top Correlations
age ↔ income: 0.72

⚠️ Data Warnings

customer_id looks like an ID column

income has 52% missing values

Potential Feature Importance
age: 0.41 → strong predictive signal
tenure: 0.32 → strong predictive signal


---

## 🧠 What Makes It Stand Out

- Automatically identifies **useful vs irrelevant features**
- No manual preprocessing required
- Mimics real-world **data analyst reasoning**
- Built using a **modular agent-based system**

---

## ⚡ Performance

- Parquet-based storage for faster I/O
- Sampling strategy for large datasets

---

## 🛠️ System Design

- Command handler
- Dataset registry
- Modular agents (AnalysisAgent, etc.)
- Logger integration

---

## 📦 Datasets

- Titanic
- Customer Churn
- Credit Card Fraud

---

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Parquet  

---

## 🚀 Future Enhancements

- RAG-based EDA advisor  
- SQL query assistant  
- Model training pipeline  
