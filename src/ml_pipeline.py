"""
ML Pipeline for Customer Segmentation

Responsibilities:
1. Load trained scaler and K-Means model
2. Predict a single customer
3. Predict customers from an uploaded DataFrame
4. Calculate dynamic dashboard KPIs

This module is designed to be imported by Streamlit.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

# Project root:
# dataset_training/
# ├── models/
# ├── data/
# └── src/
#
# Since this file is inside src/, parent.parent = project root.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

SCALER_PATH = MODEL_DIR / "scaler.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_model.pkl"


# ============================================================
# REQUIRED ML FEATURES
# ============================================================

ML_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
]


# ============================================================
# BUSINESS SEGMENT MAPPING
# ============================================================

# K-Means cluster IDs are converted into
# business-friendly segment names.

SEGMENT_MAPPING = {
    1: {
        "name": "High-Value Customers",
        "description": (
            "High-spending, frequent repeat customers "
            "with strong engagement."
        ),
    },

    3: {
        "name": "Promising Customers",
        "description": (
            "Promising buyers showing early signs of engagement, not yet loyal."
        ),
    },

    0: {
        "name": "At-Risk Customers",
        "description": (
            "Customers showing signs of disengagement "
            "or poor experience."
        ),
    },

    2: {
        "name": "Churned/Lost Customers",
        "description": (
            "Long-inactive customers unlikely to return without a strong win-back push."
        ),
    },
}


# ============================================================
# MODEL LOADING
# ============================================================

def _load_models():
    """
    Load the trained scaler and K-Means model.

    Returns:
        scaler
        kmeans_model

    Raises:
        FileNotFoundError if the .pkl files do not exist.
    """

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}\n\n"
            "Make sure scaler.pkl exists inside the models folder."
        )

    if not KMEANS_PATH.exists():
        raise FileNotFoundError(
            f"K-Means model not found:\n{KMEANS_PATH}\n\n"
            "Make sure kmeans_model.pkl exists inside the models folder."
        )

    scaler = joblib.load(SCALER_PATH)
    kmeans_model = joblib.load(KMEANS_PATH)

    return scaler, kmeans_model


# ============================================================
# DATA VALIDATION
# ============================================================

def _validate_input_columns(df: pd.DataFrame) -> None:
    """
    Check whether the DataFrame contains all six
    required ML features.
    """

    missing_columns = [
        column
        for column in ML_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input data is missing required ML columns:\n"
            f"{missing_columns}\n\n"
            f"Required columns are:\n{ML_FEATURES}"
        )


def _validate_numeric_features(df: pd.DataFrame) -> None:
    """
    Check that all ML features contain numeric values
    and do not contain NaN or infinite values.
    """

    for column in ML_FEATURES:

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if converted.isna().any():
            bad_count = converted.isna().sum()

            raise ValueError(
                f"Column '{column}' contains "
                f"{bad_count} missing/non-numeric value(s). "
                "Please clean the data before prediction."
            )

        if np.isinf(converted.to_numpy()).any():

            raise ValueError(
                f"Column '{column}' contains infinite values."
            )

        df[column] = converted
# ============================================================
# MODEL FEATURE PREPROCESSING
# ============================================================

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the exact same feature transformations used during training.

    Recency:
        kept in its original scale

    Frequency:
        log1p transformation

    Monetary:
        log1p transformation
    """

    features = df[ML_FEATURES].copy()

    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])

    return features


# ============================================================
# SEGMENT INFORMATION
# ============================================================

def _get_segment_info(cluster_id: int) -> Dict[str, str]:
    """
    Convert numeric K-Means cluster ID into
    business-friendly information.
    """

    if cluster_id not in SEGMENT_MAPPING:

        return {
            "name": f"Cluster {cluster_id}",
            "description": (
                "Cluster produced by the trained K-Means model."
            ),
        }

    return SEGMENT_MAPPING[cluster_id]


# ============================================================
# 1. SINGLE CUSTOMER INFERENCE
# ============================================================


# ============================================================
# 1. SINGLE CUSTOMER INFERENCE
# ============================================================

def predict_single_customer(
    recency: float,
    frequency: int,
    monetary: float
) -> Dict[str, Any]:
    """
    Predict the segment for one customer using
    the exact preprocessing used during training.
    """

    scaler, kmeans_model = _load_models()

    customer_df = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
    }])

    _validate_numeric_features(customer_df)

    # Apply training-time transformations
    prepared_features = _prepare_features(customer_df)

    # Preserve feature names to avoid sklearn warnings
    scaled_customer = pd.DataFrame(
        scaler.transform(prepared_features),
        columns=ML_FEATURES
    )

    cluster_id = int(
        kmeans_model.predict(scaled_customer)[0]
    )

    segment_info = _get_segment_info(cluster_id)

    return {
        "cluster_id": cluster_id,
        "segment_name": segment_info["name"],
        "description": segment_info["description"],
    }

# ============================================================
# 2. BATCH CSV / DATAFRAME PREDICTION
# ============================================================

# ============================================================
# 2. BATCH CSV / DATAFRAME PREDICTION
# ============================================================

def batch_predict_csv(
    input_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict customer segments for an entire DataFrame.

    Required ML columns:
        Recency
        Frequency
        Monetary

    Returns the original DataFrame plus:
        Predicted_Cluster_ID
        Segment_Name
        Segment
    """

    if not isinstance(input_df, pd.DataFrame):
        raise TypeError(
            "input_df must be a pandas DataFrame."
        )

    if input_df.empty:
        raise ValueError(
            "The uploaded DataFrame is empty."
        )

    scaler, kmeans_model = _load_models()

    _validate_input_columns(input_df)
    _validate_numeric_features(input_df)

    result_df = input_df.copy()

    # Apply the exact same transformations used during training
    prepared_features = _prepare_features(result_df)

    # Scale using the trained scaler
    scaled_data = pd.DataFrame(
        scaler.transform(prepared_features),
        columns=ML_FEATURES,
        index=result_df.index
    )

    # Predict clusters
    predictions = kmeans_model.predict(scaled_data)

    result_df["Predicted_Cluster_ID"] = (
        predictions.astype(int)
    )

    # Business segment name
    result_df["Segment_Name"] = (
        result_df["Predicted_Cluster_ID"]
        .map({
            cluster_id: info["name"]
            for cluster_id, info in SEGMENT_MAPPING.items()
        })
        .fillna(
            result_df["Predicted_Cluster_ID"]
            .apply(lambda x: f"Cluster {x}")
        )
    )

    return result_df


# ============================================================
# 3. DYNAMIC KPI AGGREGATOR
# ============================================================

def get_dashboard_kpis(
    predicted_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate dashboard metrics from the DataFrame
    generated by batch_predict_csv().

    Returns:
        total_customers
        average_monetary
        segment_distribution
        segment_summary
    """

    if not isinstance(predicted_df, pd.DataFrame):

        raise TypeError(
            "predicted_df must be a pandas DataFrame."
        )

    if predicted_df.empty:

        raise ValueError(
            "Cannot calculate KPIs from an empty DataFrame."
        )

    # Required columns
    required_columns = ML_FEATURES + [
        "Predicted_Cluster_ID",
        "Segment_Name",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predicted_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "predicted_df is missing required columns:\n"
            f"{missing_columns}\n\n"
            "Run batch_predict_csv() first."
        )

    # Total customers
    total_customers = len(predicted_df)

    # Average monetary value
    average_monetary = float(
        predicted_df["Monetary"].mean()
    )

    # Segment distribution
    segment_counts = (
        predicted_df["Segment_Name"]
        .value_counts()
    )

    segment_distribution = {}

    for segment_name, count in segment_counts.items():

        percentage = (
            count / total_customers
        ) * 100

        segment_distribution[segment_name] = {
            "count": int(count),
            "percentage": round(
                float(percentage),
                2
            ),
        }

    # Segment-level RFM summary
    segment_summary = (
        predicted_df
        .groupby("Segment_Name")[
            [
                "Recency",
                "Frequency",
                "Monetary",
            ]
        ]
        .mean()
        .round(2)
        .reset_index()
    )

    return {
        "total_customers": total_customers,
        "average_monetary": average_monetary,
        "segment_distribution": segment_distribution,
        "segment_summary": segment_summary,
    }