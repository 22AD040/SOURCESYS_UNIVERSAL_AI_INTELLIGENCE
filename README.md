# 🧠 SOURCESYS_GENAI_HR_INTELLIGENCE

## 🚀 GenAI-Powered HR Intelligence & Retention Prediction System

An advanced AI-driven HR Analytics Dashboard built using **Streamlit, NumPy, Pandas, Matplotlib, and Machine Learning** to simulate intelligent resume screening, bias-aware analysis, and employee retention risk prediction.

This system provides interactive data filtering, predictive modeling, and workforce intelligence insights for modern HR decision-making.

---

## 🎯 Project Objective

To build a GenAI-inspired HR Intelligence system that:

- Screens candidates using analytical filters
- Computes AI-based retention risk scores
- Detects workforce trends and bias patterns
- Predicts job change probability using Logistic Regression
- Provides interactive data visualization
- Enables filtered CSV data download

---

## 🏗️ Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn (Logistic Regression)**
- **Virtual Environment (venv)**

---

## 📊 Dataset Used

HR Analytics – Job Change of Data Scientists  
🔗 https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists

Dataset includes:

- Training hours
- Education level
- Experience
- Company size
- City development index
- Target variable (0 = Stay, 1 = Leave)

---

## 💡 Key Features

### ✅ Interactive Filtering
- Numeric range filtering
- Education-level selection
- Dynamic data updates

### ✅ AI Retention Risk Score
Custom risk scoring logic based on:
- Training hours
- Development index

Risk Categories:
- Low
- Medium
- High

### ✅ Machine Learning Prediction
- Logistic Regression model
- Feature encoding
- Train/Test split
- Model accuracy evaluation

### ✅ Advanced Visualizations
- Line Chart
- Bar Chart
- Histogram
- Scatter Plot
- Correlation Heatmap
- Target Distribution Chart

### ✅ HR Metrics Dashboard
- Total Candidates
- Average Training Hours
- Risk Score
- Job Change %

### ✅ CSV Export Feature
Download filtered data using:

```python
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download filtered data",
    csv,
    "filtered_data.csv",
    "text/csv"
)