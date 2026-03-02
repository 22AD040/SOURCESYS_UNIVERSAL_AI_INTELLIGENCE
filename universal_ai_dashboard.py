import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score

st.set_page_config(page_title="Universal AI Intelligence System", layout="wide")

st.title("🧠 Universal AI Data Intelligence & Predictive Analytics")
st.markdown("Upload any CSV file to perform dynamic analysis and machine learning.")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_columns) == 0:
        st.warning("No numeric columns found in dataset.")
        st.stop()

    selected_numeric = st.sidebar.selectbox("Select Numeric Column for Filtering", numeric_columns)

    min_val = float(df[selected_numeric].min())
    max_val = float(df[selected_numeric].max())

    range_values = st.sidebar.slider("Select Range", min_val, max_val, (min_val, max_val))

    filtered_df = df[
        (df[selected_numeric] >= range_values[0]) &
        (df[selected_numeric] <= range_values[1])
    ].copy()

    filtered_df.fillna(filtered_df.mean(numeric_only=True), inplace=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", len(filtered_df))

    with col2:
        st.metric("Mean Value", round(filtered_df[selected_numeric].mean(), 2))

    with col3:
        st.metric("Standard Deviation", round(filtered_df[selected_numeric].std(), 2))

    st.subheader("📊 Filtered Data")
    st.dataframe(filtered_df)

    st.subheader("📈 Correlation Heatmap")

    corr = filtered_df[numeric_columns].corr()

    fig, ax = plt.subplots()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=90)
    ax.set_yticklabels(numeric_columns)
    st.pyplot(fig)

    st.subheader("📉 Visualization")

    chart_type = st.radio("Select Chart Type", ["Line", "Bar", "Histogram", "Scatter"])

    if chart_type == "Line":
        st.line_chart(filtered_df[selected_numeric])

    elif chart_type == "Bar":
        st.bar_chart(filtered_df[selected_numeric])

    elif chart_type == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(filtered_df[selected_numeric], bins=20)
        st.pyplot(fig)

    else:
        x_axis = st.selectbox("Select X Axis", numeric_columns)
        y_axis = st.selectbox("Select Y Axis", numeric_columns)

        fig, ax = plt.subplots()
        ax.scatter(filtered_df[x_axis], filtered_df[y_axis])
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        st.pyplot(fig)

    st.subheader("🤖 Machine Learning Module")

    target_column = st.selectbox("Select Target Column", df.columns)

    feature_columns = st.multiselect("Select Feature Columns", numeric_columns)

    if target_column and len(feature_columns) > 0:

        df_ml = df.copy()
        df_ml.fillna(df_ml.mean(numeric_only=True), inplace=True)

        le = LabelEncoder()
        for col in df_ml.select_dtypes(include="object").columns:
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        X = df_ml[feature_columns]
        y = df_ml[target_column]

        if y.nunique() <= 10:
            model_type = "Classification"
        else:
            model_type = "Regression"

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_type == "Classification":
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            st.success(f"Classification Accuracy: {round(accuracy * 100, 2)}%")

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax)
            st.pyplot(fig)

        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            st.success(f"Regression R² Score: {round(r2, 3)}")

    csv = filtered_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        "Download Filtered Data",
        csv,
        "filtered_data.csv",
        "text/csv"
    )

else:
    st.info("Upload a CSV file to start")