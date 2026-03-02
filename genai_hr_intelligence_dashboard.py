import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="GenAI HR Intelligence", layout="wide")

st.title("🧠 GenAI Resume Screening & Predictive Retention Intelligence")
st.markdown("AI-powered HR Analytics with Risk Prediction & Bias Insights")

uploaded_file = st.sidebar.file_uploader("Upload HR Analytics CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.sidebar.header("🔎 Filter Section")

    numeric_columns = df.select_dtypes(include=np.number).columns
    selected_numeric = st.sidebar.selectbox(
        "Select Numeric Column",
        numeric_columns
    )

    min_val = float(df[selected_numeric].min())
    max_val = float(df[selected_numeric].max())

    range_values = st.sidebar.slider(
        "Select Range",
        min_val,
        max_val,
        (min_val, max_val)
    )

    filtered_df = df[
        (df[selected_numeric] >= range_values[0]) &
        (df[selected_numeric] <= range_values[1])
    ]

    # -------------------------
    # AI Retention Risk Score
    # -------------------------
    filtered_df["Retention_Risk_Score"] = (
        0.6 * (filtered_df["training_hours"] /
               filtered_df["training_hours"].max())
        + 0.4 * filtered_df["city_development_index"]
    )

    # Risk Category
    filtered_df["Risk_Category"] = np.where(
        filtered_df["Retention_Risk_Score"] > 0.7,
        "High",
        np.where(
            filtered_df["Retention_Risk_Score"] > 0.4,
            "Medium",
            "Low"
        )
    )

    # -------------------------
    # Metrics Section
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Candidates", len(filtered_df))

    with col2:
        st.metric("Avg Training Hours",
                  round(np.mean(filtered_df["training_hours"]), 2))

    with col3:
        st.metric("Avg Risk Score",
                  round(np.mean(filtered_df["Retention_Risk_Score"]), 2))

    with col4:
        if "target" in filtered_df.columns:
            st.metric("Job Change %",
                      round(np.mean(filtered_df["target"]) * 100, 2))

    st.subheader("📊 Filtered Data")
    st.dataframe(filtered_df)

    # -------------------------
    # Correlation Heatmap
    # -------------------------
    st.subheader("📈 Correlation Analysis")

    corr = filtered_df[numeric_columns].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=90)
    ax.set_yticklabels(numeric_columns)
    st.pyplot(fig)

    # -------------------------
    # ML Prediction Section
    # -------------------------
    if "target" in df.columns:

        st.subheader("🤖 ML Retention Prediction")

        df_ml = df.copy()

        # Encode categorical columns
        le = LabelEncoder()
        for col in df_ml.select_dtypes(include="object").columns:
            df_ml[col] = df_ml[col].astype(str)
            df_ml[col] = le.fit_transform(df_ml[col])

        X = df_ml.drop("target", axis=1)
        y = df_ml["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    # -------------------------
    # Chart Section
    # -------------------------
    chart_type = st.radio(
        "Select Chart Type",
        ["Line", "Bar", "Histogram", "Scatter"]
    )

    if chart_type == "Line":
        st.line_chart(filtered_df[selected_numeric])

    elif chart_type == "Bar":
        st.bar_chart(filtered_df[selected_numeric])

    elif chart_type == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(filtered_df[selected_numeric], bins=20)
        st.pyplot(fig)

    else:
        fig, ax = plt.subplots()
        ax.scatter(
            filtered_df["training_hours"],
            filtered_df["Retention_Risk_Score"]
        )
        ax.set_xlabel("Training Hours")
        ax.set_ylabel("Risk Score")
        st.pyplot(fig)

    # -------------------------
    # CSV Download
    # -------------------------
    csv = filtered_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download filtered data",
        csv,
        "filtered_data.csv",
        "text/csv"
    )

else:
    st.info("upload a csv file to start")