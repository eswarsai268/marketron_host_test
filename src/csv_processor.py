import pandas as pd
GBP_TO_INR = 105

class CSVProcessorError(Exception):
    pass

# ============================================================
# FUNCTIONS: LOAD & PROFILE
# ============================================================

def load_csv(file_source) -> pd.DataFrame:
    """Read an uploaded CSV file safely with encoding fallbacks."""
    try:
        df = pd.read_csv(file_source)
    except UnicodeDecodeError:
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        try:
            df = pd.read_csv(file_source, encoding="latin1")
        except Exception as exc:
            raise CSVProcessorError("The CSV text encoding could not be processed.") from exc
    except Exception as exc:
        raise CSVProcessorError(f"The uploaded file could not be read: {exc}") from exc

    # Remove completely empty rows and columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    df = df.reset_index(drop=True)

    if df.empty:
        raise CSVProcessorError("The CSV contains no usable records.")

    return df


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Analyze the uploaded CSV for UI feedback."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
    }

# ============================================================
# BLOCK 1: THE ROUTER & PATH A
# ============================================================

def process_mapped_data(df: pd.DataFrame, mapping_mode: str, column_map: dict) -> pd.DataFrame:
    """
    Routes the uploaded dataframe through the correct processing pipeline 
    based on the user's UI selection.
    
    mapping_mode: 'direct_rfm' OR 'raw_transactions'
    column_map: Dictionary of how the user mapped their columns
    """
    if mapping_mode == "direct_rfm":
        return _process_direct_rfm(df, column_map)
        
    elif mapping_mode == "raw_transactions":
        return _process_raw_transactions(df, column_map)
        
    else:
        raise CSVProcessorError("Invalid processing mode selected.")


def _process_direct_rfm(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """
    Path A: User provides pre-calculated RFM values.
    Monetary is expected in GBP and is converted to INR.
    """
    try:
        mapped_df = df.rename(columns={
            column_map["customer_id"]: "CustomerID",  # 👈 Added Customer ID mapping
            column_map["recency"]: "Recency",
            column_map["frequency"]: "Frequency",
            column_map["monetary"]: "Monetary"
        })

        # 👈 Added CustomerID to required list
        required = ["CustomerID", "Recency", "Frequency", "Monetary"]
        missing = [col for col in required if col not in mapped_df.columns]

        if missing:
            raise CSVProcessorError(
                f"Missing required columns after mapping: {missing}"
            )

        # Ensure Monetary is numeric
        mapped_df["Monetary"] = pd.to_numeric(mapped_df["Monetary"], errors="coerce")

        if mapped_df["Monetary"].isna().any():
            raise CSVProcessorError("Monetary column contains missing/non-numeric values.")

        # Convert GBP → INR to match training
        mapped_df["Monetary"] = mapped_df["Monetary"] * GBP_TO_INR

        return mapped_df

    except KeyError as e:
        raise CSVProcessorError(f"Mapping configuration is missing a required field: {e}")

# ============================================================
# BLOCK 2: RAW TRANSACTION AGGREGATION (PATH B)
# ============================================================

def _process_raw_transactions(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """
    Path B: The user uploaded raw transactions. 
    We group the data by customer and calculate Recency, Frequency, and Monetary dynamically.
    """
    try:
        # 1. Isolate the exact columns we need using the user's mapping
        temp_df = df[[
            column_map["customer_id"], 
            column_map["order_date"], 
            column_map["order_id"], 
            column_map["spend"]
        ]].copy()
        
        # Standardize temporary column names for easy math
        temp_df.columns = ["CustomerID", "OrderDate", "OrderID", "Spend"]
        
        # 2. Convert to datetime (coerce bad formats to NaT, then drop those broken rows safely)
        temp_df["OrderDate"] = pd.to_datetime(temp_df["OrderDate"], errors="coerce")
        temp_df = temp_df.dropna(subset=["OrderDate"])
        
        # 3. Ensure spend is strictly numeric
        temp_df["Spend"] = pd.to_numeric(temp_df["Spend"], errors="coerce").fillna(0)

        temp_df["Spend"] = temp_df["Spend"] * GBP_TO_INR
        
        # 4. Calculate Snapshot Date (most recent transaction in the entire CSV + 1 day)
        snapshot_date = temp_df["OrderDate"].max() + pd.Timedelta(days=1)
        
        # 5. Group by Customer and execute the RFM math
        rfm_df = temp_df.groupby("CustomerID").agg({
            "OrderDate": lambda x: (snapshot_date - x.max()).days,  # Days since last order
            "OrderID": "nunique",                                   # Count of distinct orders
            "Spend": "sum"                                          # Total money spent
        }).reset_index()
        
        # 6. Rename exactly to our ML pipeline schema
        rfm_df.rename(columns={
            "OrderDate": "Recency",
            "OrderID": "Frequency",
            "Spend": "Monetary"
        }, inplace=True)
        
        # Note: The output dataframe will now contain CustomerID, Recency, Frequency, and Monetary.
        # Your ml_pipeline.py will perfectly ignore CustomerID during scaling, but keep it for the dashboard!
        
        return rfm_df
        
    except KeyError as e:
        raise CSVProcessorError(f"Mapping configuration is missing a required field: {e}")
    except Exception as e:
        raise CSVProcessorError(f"Failed to calculate RFM from raw transactions: {e}")