# 🧠 SOURCESYS_GENAI_HR_INTELLIGENCE

## 🚀 Universal AI Data Intelligence & Predictive Analytics Platform

An enterprise-ready AI-powered analytics dashboard built using **Streamlit, NumPy, Pandas, Matplotlib, and Machine Learning**.

This system dynamically adapts to any uploaded CSV dataset and automatically performs data analysis, visualization, and predictive modeling without hardcoded dependencies.

---

## 🎯 Project Objective

To build a fully dynamic AI analytics system that:

- Accepts any CSV dataset
- Performs intelligent data filtering
- Generates statistical insights
- Automatically detects ML problem type
- Executes Classification or Regression models
- Prevents dataset-specific errors
- Enables downloadable filtered data

---

## 🏗️ Tech Stack

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Virtual Environment (venv)

---

## 📂 Features

### ✅ Dynamic Dataset Handling
- Works with ANY CSV file
- No hardcoded column dependencies
- Automatic numeric column detection

### ✅ Smart Data Filtering
- Interactive range-based filtering
- Real-time metric updates
- Statistical summaries

### ✅ Advanced Visualization
- Line Chart
- Bar Chart
- Histogram
- Scatter Plot
- Correlation Heatmap

### ✅ Intelligent Machine Learning Module

The system automatically detects:

- Classification problems (≤ 10 unique target values)
- Regression problems (> 10 unique target values)

#### Classification
- Logistic Regression
- Accuracy Score
- Confusion Matrix

#### Regression
- Linear Regression
- R² Score

---

## 📥 CSV Export Feature

Download filtered dataset using:

```python
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Filtered Data",
    csv,
    "filtered_data.csv",
    "text/csv"
)