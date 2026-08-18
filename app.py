"""
Streamlit app — Adult Income Classification (BITS ML Assignment 2).
Upload test CSV, select a trained model, view metrics and confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
]


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    models = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_OPTIONS.items()
    }
    return preprocessor, label_encoder, models


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    df = df.copy()
    # Allow either income or Income as target
    target_col = None
    for cand in ("income", "Income", "target", "class"):
        if cand in df.columns:
            target_col = cand
            break

    y = None
    if target_col is not None:
        y = df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
        df = df.drop(columns=[target_col])

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing)
            + ". Use the provided test_data.csv format."
        )

    return df[FEATURE_COLUMNS], y


def main() -> None:
    st.set_page_config(
        page_title="Adult Income Classifier",
        page_icon="📊",
        layout="wide",
    )

    st.title("Adult Income Classification")
    st.markdown(
        """
        **BITS WILP — Machine Learning Assignment 2**

        Predict whether annual income is **<=50K** or **>50K** using five classical
        classifiers trained on the UCI Adult Income dataset.
        """
    )

    if not (MODEL_DIR / "preprocessor.pkl").exists():
        st.error(
            "Model artifacts not found. Open `ml_pipeline.ipynb` and Run All (or run `python train.py`)."
        )
        st.stop()

    preprocessor, label_encoder, models = load_artifacts()

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
        model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))
        st.caption(
            "Tip: upload the repo file `test_data.csv`, or any CSV with the same columns."
        )

    st.subheader("1. Dataset upload")
    if uploaded is None:
        st.info("Upload a CSV with features (and ideally an `income` label column) to begin.")
        sample = ROOT / "test_data.csv"
        if sample.exists():
            st.download_button(
                "Download bundled test_data.csv",
                data=sample.read_bytes(),
                file_name="test_data.csv",
                mime="text/csv",
            )
        st.stop()

    try:
        raw_df = pd.read_csv(uploaded)
        X, y_text = prepare_frame(raw_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read / validate CSV: {exc}")
        st.stop()

    st.write(f"Loaded **{len(X)}** rows × **{X.shape[1]}** features.")
    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    if y_text is None:
        st.warning(
            "No `income` column found — predictions will be shown without evaluation metrics."
        )

    st.subheader(f"2. Model: {model_name}")
    model = models[model_name]

    try:
        X_t = preprocessor.transform(X)
        y_pred = model.predict(X_t)
        y_proba = model.predict_proba(X_t)[:, 1]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Inference failed: {exc}")
        st.stop()

    pred_labels = label_encoder.inverse_transform(y_pred)
    result_df = X.copy()
    result_df["predicted_income"] = pred_labels
    result_df["prob_>50K"] = y_proba.round(4)

    st.subheader("3. Predictions (sample)")
    st.dataframe(result_df.head(25), use_container_width=True)

    if y_text is not None:
        # Align label strings with training encoder classes
        try:
            y_true = label_encoder.transform(y_text)
        except ValueError as exc:
            st.error(
                f"Target labels in CSV do not match training classes "
                f"{list(label_encoder.classes_)}: {exc}"
            )
            st.stop()

        metrics = compute_metrics(y_true, y_pred, y_proba)
        st.subheader("4. Evaluation metrics")
        metric_df = pd.DataFrame(
            [{"Metric": k, "Value": f"{v:.4f}"} for k, v in metrics.items()]
        )
        cols = st.columns(3)
        for i, (name, value) in enumerate(metrics.items()):
            cols[i % 3].metric(name, f"{value:.4f}")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

        st.subheader("5. Confusion matrix & classification report")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig)
        plt.close(fig)

        report = classification_report(
            y_true, y_pred, target_names=label_encoder.classes_, output_dict=False
        )
        st.code(report, language="text")

    st.markdown("---")
    st.caption(
        "Models and preprocessors were fit on the Adult Income training split only "
        "(One-Hot encoding + StandardScaler). Artifacts live in the `model/` folder."
    )


if __name__ == "__main__":
    main()
