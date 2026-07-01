import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)


@st.cache_data
def load_data(path="Dataset/Salary_Data.csv"):
    return pd.read_csv(path)


@st.cache_resource
def train_linear_model(df):
    data = df.copy()

    if "Salary" not in data.columns:
        raise ValueError("The column 'Salary' does not exist in the dataset.")

    data = data.dropna(subset=["Salary"])

    X = data.drop("Salary", axis=1)
    y = data["Salary"]

    categorical_cols = [c for c in ["Gender", "Education Level", "Job Title"] if c in X.columns]
    numeric_cols = [c for c in ["Age", "Years of Experience"] if c in X.columns]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean"))
            ]), numeric_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression())
    ])

    if len(X) < 2:
        raise ValueError("Dataset is too small after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    return pipeline, (X_test, y_test, y_pred), metrics, categorical_cols, numeric_cols


@st.cache_resource
def train_logistic_model(df, threshold=70000):
    data = df.copy()

    if "Salary" not in data.columns:
        raise ValueError("The column 'Salary' does not exist in the dataset.")

    data["Salary_Class"] = (data["Salary"] >= threshold).astype(int)

    numeric_cols = [c for c in ["Age", "Years of Experience"] if c in data.columns]
    cat_cols = [c for c in ["Gender", "Education Level", "Job Title"] if c in data.columns]

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    for col in cat_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    selected_features = [
        "Age", "Gender", "Education Level",
        "Job Title", "Years of Experience"
    ]
    selected_features = [f for f in selected_features if f in data.columns]

    X = data[selected_features]
    y = data["Salary_Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)
    y_proba = log_model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba)
    }

    return (
        log_model, scaler, encoders, selected_features,
        (X_test, y_test, y_pred, y_proba, cm, fpr, tpr),
        metrics, numeric_cols, cat_cols
    )


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label"
    )

    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    st.pyplot(fig)


def plot_roc_curve(fpr, tpr, auc_score):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Logistic Regression")
    ax.legend(loc="lower right")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Salary ML Project", layout="wide")

    st.title("Machine Learning Salary Prediction and Classification")

    st.markdown("""
    This web application demonstrates two machine learning models:

    1. **Linear Regression:** Predicts the actual salary value.
    2. **Logistic Regression:** Classifies salary as High or Low based on a threshold of 70,000.

    Both models use the same dataset and preprocessing steps.
    """)

    df = load_data()

    tab1, tab2, tab3 = st.tabs([
        "Dataset Overview",
        "Linear Regression (Salary Prediction)",
        "Logistic Regression (Salary Classification)"
    ])

    with tab1:
        st.header("Dataset Overview")
        st.write("First 10 rows of the dataset:")
        st.dataframe(df.head(10))

        st.write("Statistical summary:")
        st.write(df.describe())

        st.write("Columns:")
        st.write(list(df.columns))

    with tab2:
        st.header("Linear Regression - Salary Prediction")

        try:
            lin_model, (_, y_test_lin, y_pred_lin), lin_metrics, cat_cols_lin, num_cols_lin = train_linear_model(df)

            col_metrics, col_plot = st.columns(2)

            with col_metrics:
                st.subheader("Model Metrics")
                st.write(f"Mean Absolute Error (MAE): {lin_metrics['MAE']:.2f}")
                st.write(f"R² Score: {lin_metrics['R2']:.3f}")

            with col_plot:
                st.subheader("Actual vs Predicted Salary")
                fig, ax = plt.subplots()
                ax.scatter(y_test_lin, y_pred_lin)
                ax.set_xlabel("Actual Salary")
                ax.set_ylabel("Predicted Salary")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("Predict Salary for a New Input")

            user_input = {}

            for col in num_cols_lin:
                default_val = float(df[col].median())
                user_input[col] = st.number_input(f"{col}", value=default_val)

            for col in cat_cols_lin:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                user_input[col] = st.selectbox(f"{col}", options=options)

            if st.button("Predict Salary"):
                user_df = pd.DataFrame([user_input])
                pred_salary = lin_model.predict(user_df)[0]
                st.success(f"Predicted Salary: {pred_salary:,.2f}")

        except Exception as e:
            st.error(f"Error in Linear Regression: {e}")

    with tab3:
        st.header("Logistic Regression - Salary Classification")

        try:
            (
                log_model,
                scaler,
                encoders,
                selected_features,
                (_, _, _, _, cm, fpr, tpr),
                log_metrics,
                numeric_cols_log,
                cat_cols_log
            ) = train_logistic_model(df)

            st.subheader("Model Metrics")
            st.write(f"Accuracy: {log_metrics['Accuracy']:.3f}")
            st.write(f"Precision: {log_metrics['Precision']:.3f}")
            st.write(f"Recall: {log_metrics['Recall']:.3f}")
            st.write(f"F1-score: {log_metrics['F1-score']:.3f}")
            st.write(f"AUC-ROC: {log_metrics['AUC-ROC']:.3f}")

            st.markdown("---")
            col_cm, col_roc = st.columns(2)

            with col_cm:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(cm, ["Low Salary", "High Salary"])

            with col_roc:
                st.subheader("ROC Curve")
                plot_roc_curve(fpr, tpr, log_metrics["AUC-ROC"])

            st.markdown("---")
            st.subheader("Classify New Input")

            user_input_log = {}

            for col in selected_features:
                if col in numeric_cols_log:
                    default_val = float(df[col].median())
                    user_input_log[col] = st.number_input(f"{col}", value=default_val, key=f"log_{col}")
                elif col in cat_cols_log:
                    options = sorted(df[col].dropna().astype(str).unique().tolist())
                    user_input_log[col] = st.selectbox(f"{col}", options=options, key=f"log_{col}")

            if st.button("Classify Salary"):
                user_df_log = pd.DataFrame([user_input_log])

                for col in cat_cols_log:
                    le = encoders[col]
                    val = str(user_df_log[col][0])

                    if val not in le.classes_:
                        le.classes_ = np.append(le.classes_, val)

                    user_df_log[col] = le.transform([val])[0]

                user_X = user_df_log[selected_features]
                user_X_scaled = scaler.transform(user_X)

                proba = log_model.predict_proba(user_X_scaled)[0][1]
                pred_class = 1 if proba >= 0.5 else 0

                label = "High Salary (>= 70,000)" if pred_class == 1 else "Low Salary (< 70,000)"
                st.success(f"Prediction: {label}")
                st.info(f"Probability of High Salary: {proba:.2f}")

        except Exception as e:
            st.error(f"Error in Logistic Regression: {e}")


if __name__ == "__main__":
    main()