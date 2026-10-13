from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
import pandas as pd
from xgboost import XGBRegressor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

_filepath=config["data"]["path"]
mcp = FastMCP("PricingServer")

def load_data():
    try:
        if _filepath.lower().endswith('.csv'):
            df = pd.read_csv(_filepath)
        else:
            df = pd.read_excel(_filepath)
            
        if not df.empty:
            print(f'Loaded dataframe with {len(df)} rows')
            return df  # Added return statement so you can use the data
        else:
            raise ValueError(f'Empty dataframe found at {_filepath}')
            
    except Exception as e:
        raise ValueError(f'Error: {str(e)}')
           
@mcp.tool()
def calculate_price_elasticity(
    upc: int
) -> dict:
    """
    Calculate Price Elasticity of Demand (PED) for a UPC using
    multivariate log-log linear regression.

    Model:
        log(UNITS) =
            beta_0
            + beta_1 * log(PRICE)
            + beta_2 * FEATURE
            + beta_3 * DISPLAY
            + beta_4 * TPR_ONLY

    Args:
        csv_path: Path to the retail CSV dataset.
        upc: UPC/product identifier.

    Returns:
        Dictionary containing PED, R-squared, promotion effects,
        sample size, and average price.
    """

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    try:
        df = load_data()
    except Exception as e:
        return {
            "error": f"Unable to read CSV file: {str(e)}"
        }

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    required_columns = [
        "UPC",
        "UNITS",
        "PRICE"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        return {
            "error": f"Missing required columns: {missing_columns}"
        }

    # ---------------------------------------------------------
    # Filter UPC
    # ---------------------------------------------------------
    product_data = df[df["UPC"] == upc].copy()

    if product_data.empty:
        return {
            "error": f"No observations found for UPC {upc}"
        }

    # ---------------------------------------------------------
    # Clean data
    # ---------------------------------------------------------
    product_data["UNITS"] = pd.to_numeric(
        product_data["UNITS"],
        errors="coerce"
    )

    product_data["PRICE"] = pd.to_numeric(
        product_data["PRICE"],
        errors="coerce"
    )

    product_data = product_data.dropna(
        subset=["UNITS", "PRICE"]
    )

    # Log transformation requires positive values
    product_data = product_data[
        (product_data["UNITS"] > 0) &
        (product_data["PRICE"] > 0)
    ]

    if len(product_data) < 5:
        return {
            "error": (
                "Insufficient valid observations. "
                "At least 5 positive price/unit observations are required."
            )
        }

    # ---------------------------------------------------------
    # Log transformation
    # ---------------------------------------------------------
    product_data["log_units"] = np.log(
        product_data["UNITS"]
    )

    product_data["log_price"] = np.log(
        product_data["PRICE"]
    )

    # ---------------------------------------------------------
    # Feature selection
    # ---------------------------------------------------------
    feature_cols = ["log_price"]

    for col in ["FEATURE", "DISPLAY", "TPR_ONLY"]:

        if col in product_data.columns:

            product_data[col] = pd.to_numeric(
                product_data[col],
                errors="coerce"
            ).fillna(0)

            # Only include variables with variation
            if product_data[col].nunique() > 1:
                feature_cols.append(col)

    # ---------------------------------------------------------
    # Prepare X and y
    # ---------------------------------------------------------
    X = product_data[feature_cols]
    y = product_data["log_units"]

    # ---------------------------------------------------------
    # Fit regression
    # ---------------------------------------------------------
    model = LinearRegression()

    try:
        model.fit(X, y)
    except Exception as e:
        return {
            "error": f"Regression failed: {str(e)}"
        }

    # ---------------------------------------------------------
    # Price elasticity
    # ---------------------------------------------------------
    price_coeff_index = feature_cols.index("log_price")

    ped_elasticity = model.coef_[price_coeff_index]

    r_squared = model.score(X, y)

    # ---------------------------------------------------------
    # Promotion effects
    # ---------------------------------------------------------
    promo_effects = {}

    for idx, col in enumerate(feature_cols):

        if col == "log_price":
            continue

        coefficient = model.coef_[idx]

        uplift = (
            np.exp(coefficient) - 1
        ) * 100

        promo_effects[col] = round(
            float(uplift),
            2
        )

    # ---------------------------------------------------------
    # Product name
    # ---------------------------------------------------------
    if "DESCRIPTION" in product_data.columns:
        product_name = str(
            product_data["DESCRIPTION"].iloc[0]
        )
    else:
        product_name = f"UPC {upc}"

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------
    return {
        "upc": int(upc),

        "product_name": product_name,

        "multivariate_ped_elasticity": round(
            float(ped_elasticity),
            3
        ),

        "interpretation": (
            "Elastic"
            if abs(ped_elasticity) > 1
            else "Inelastic"
        ),

        "r_squared": round(
            float(r_squared),
            3
        ),

        "sample_size": int(
            len(product_data)
        ),

        "controlling_variables": feature_cols[1:],

        "promotion_uplifts": promo_effects,

        "avg_unit_price": round(
            float(product_data["PRICE"].mean()),
            2
        )
    }

@mcp.tool()
def _forecast_demand(f_level: str, 
                     category: str,
                     N: int | None = None) -> dict:
    """
    Forecast demand for N horizons ahead for a given hierarchy level.

    Current implementation:
    - Aggregates data at the requested hierarchy level.
    - Uses the last `horizon` observations as a holdout test set.
    - Trains XGBRegressor on the historical observations.
    - Predicts the holdout horizon.
    """

    try:
        # ---------------------------------------------------------
        # 1. Load data
        # ---------------------------------------------------------
        df = load_data()

        if df.empty:
            return {"error": "Dataset is empty."}
        else:
            print(f'Loaded dataset with {len(df)} rows')

    except Exception as e:
        return {
            "error": f"Unable to read data file: {str(e)}"
        }

    try:
        # ---------------------------------------------------------
        # 2. Validate forecasting level
        # ---------------------------------------------------------
        f_level = f_level.lower().strip()

        if f_level != "category":
            return {
                "error": (
                    f"Unsupported forecasting level: '{f_level}'. "
                    "Currently supported level: 'category'."
                )
            }

        if N is not None and N <= 0:
           print(f'Warning: Invalid horizon {N}, using default from config')
           horizon = int(config["forecast"]["horizon"])
        elif N is not None:
             horizon=int(N)
        else:
            horizon = int(config["forecast"]["horizon"])
        
        # ---------------------------------------------------------
        # 3. Filter hierarchy level
        # ---------------------------------------------------------
        
        #category = config["forecast"]["name"]
        category = category.strip()

        if "CATEGORY" not in df.columns:
            return {
                "error": "CATEGORY column not found in dataset."
            }

        df = df[df["CATEGORY"].astype(str).str.strip().str.upper()== category.upper()].copy()
        
        if df.empty:
            return {
                "error": f"No data found for category: {category}"
            }

        # ---------------------------------------------------------
        # 4. Create time variable
        # ---------------------------------------------------------
        if "WEEK_END_DATE" not in df.columns:
            return {
                "error": "WEEK_END_DATE column not found in dataset."
            }

        df["time"] = pd.to_datetime(
            df["WEEK_END_DATE"],
            errors="coerce"
        )

        # Remove invalid dates
        df = df.dropna(subset=["time"])

        if df.empty:
            return {
                "error": "No valid dates found in WEEK_END_DATE."
            }

        # ---------------------------------------------------------
        # 5. Aggregate to category-week level
        # ---------------------------------------------------------
        agg_dict = {
            # Target variable
            "UNITS": "sum",

            # Price features
            "BASE_PRICE": "mean",
            "PRICE": "mean",

            # Promotional indicators
            "FEATURE": "max",
            "DISPLAY": "max",
            "TPR_ONLY": "max",
        }

        missing_agg_cols = [
            col for col in agg_dict
            if col not in df.columns
        ]

        if missing_agg_cols:
            return {
                "error": (
                    "Missing columns required for aggregation: "
                    f"{missing_agg_cols}"
                )
            }

        df = (
            df.groupby("time", as_index=False)
              .agg(agg_dict)
              .sort_values("time")
              .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # 6. Validate required columns
        # ---------------------------------------------------------
        missing_cols = [
            col
            for col in config["forecast"]["required_data"]
            if col not in df.columns
        ]

        if missing_cols:
            return {
                "error": (
                    "Missing required columns in dataset: "
                    f"{missing_cols}"
                )
            }

        # ---------------------------------------------------------
        # 7. Handle missing values
        # ---------------------------------------------------------
        strategies = config["forecast"][
            "data_preprocessing"
        ]["missing_value_strategy"]

        for k, v in strategies.items():

            if k not in df.columns:
                continue

            if v == "forward_fill":
                df[k] = df[k].ffill().bfill()

            else:
                df[k] = df[k].fillna(v)

        # ---------------------------------------------------------
        # 8. Feature Engineering
        # ---------------------------------------------------------


        df["discount_depth"] = np.where(
            df["BASE_PRICE"] > 0,
            (df["BASE_PRICE"] - df["PRICE"])
            / df["BASE_PRICE"],
            0
        )

        # Promotional indicator
        df["is_promo"] = (
            (df["FEATURE"] == 1)
            | (df["TPR_ONLY"] == 1)
            | (df["DISPLAY"] == 1)
        ).astype(int)

        # Replace infinite values
        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        # Final missing-value handling after feature engineering
        df = df.dropna(
            subset=[
                "UNITS",
                "BASE_PRICE",
                "PRICE",
                "FEATURE",
                "DISPLAY",
                "TPR_ONLY",
                "discount_depth",
                "is_promo"
            ]
        )

        # ---------------------------------------------------------
        # 9. Define features and horizon
        # ---------------------------------------------------------
        features = [
            "BASE_PRICE",
            "PRICE",
            "FEATURE",
            "DISPLAY",
            "TPR_ONLY",
            "discount_depth",
            "is_promo"
        ]

       

        target_variable = config["forecast"][
            "target_variable"
        ]

        # ---------------------------------------------------------
        # 10. Validate dataset size
        # ---------------------------------------------------------
        if len(df) <= horizon:
            return {
                "error": (
                    f"Insufficient observations. "
                    f"Need more than {horizon} rows, "
                    f"but only {len(df)} are available."
                )
            }

        # ---------------------------------------------------------
        # 11. Train / Test split
        # ---------------------------------------------------------
        X_train = df.iloc[:-horizon][features]
        y_train = df.iloc[:-horizon][target_variable]

        X_test = df.iloc[-horizon:][features]
        y_test = df.iloc[-horizon:][target_variable]

        # ---------------------------------------------------------
        # 12. Train XGBoost model
        # ---------------------------------------------------------
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror"
        )

        model.fit(X_train,y_train)

        # ---------------------------------------------------------
        # 13. Predict holdout horizon
        # ---------------------------------------------------------
        predictions = model.predict(X_test)

        # Prevent negative demand
        predictions = np.maximum(
            predictions,
            0
        )

        # ---------------------------------------------------------
        # 14. Create forecast output
        # ---------------------------------------------------------
        recent_dates = (
            df["time"]
            .tail(horizon)
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        forecast_series = [
            {
                "period": date,
                "predicted_units": round(
                    float(pred),
                    2
                )
            }
            for date, pred in zip(
                recent_dates,
                predictions
            )
        ]

        # ---------------------------------------------------------
        # 15. Calculate summary metrics
        # ---------------------------------------------------------
        avg_baseline = float(
            y_test.mean()
        )

        avg_predicted = float(
            np.mean(predictions)
        )

        if avg_baseline != 0:
            pct_change = round(
                (
                    (avg_predicted - avg_baseline)
                    / avg_baseline
                ) * 100,
                2
            )
        else:
            pct_change = None

        # ---------------------------------------------------------
        # 16. Evaluation metrics
        # ---------------------------------------------------------
        rmse = float(
            np.sqrt(
                np.mean(
                    (y_test.values - predictions) ** 2
                )
            )
        )

        mape = float(
            np.mean(
                np.abs(
                    (
                        y_test.values - predictions
                    )
                    / np.where(
                        y_test.values != 0,
                        y_test.values,
                        np.nan
                    )
                )
            ) * 100
        )

        wape= float(np.sum(np.abs(y_test.values-predictions))/np.sum(np.abs(y_test.values)))*100

        # ---------------------------------------------------------
        # 17. Return result
        # ---------------------------------------------------------
        return {
            "status": "success",
            "forecast_level": f_level,
            "category": category,
            "target_variable": target_variable,
            "forecast_horizon": horizon,
            "model_used": "XGBRegressor",

            "summary_metrics": {
                "historical_avg_units": round(
                    float(df[target_variable].mean()),
                    2
                ),
                "test_avg_units": round(
                    avg_baseline,
                    2
                ),
                "forecasted_avg_units": round(
                    avg_predicted,
                    2
                ),
                "expected_volume_change_pct": pct_change,
                "rmse": round(rmse, 2),
                "mape_pct": round(mape, 2),
                "wape_pct":round(wape,2)
            },

            "detailed_forecast": forecast_series
        }

    except Exception as e:

        return {
            "status": "error",
            "error": f"Forecasting failed: {str(e)}"
        }

@mcp.tool()
def demand_transfer(target_upc:int, price_change_pct:float,elasticity:float)-> dict:
    if target_upc is None:
        print(f'WARNING! Using fallback UPC {config["demand_transfer"]["fallback_upc"]}')

    df=load_data()
    if not df.empty:
       logging.info(f'Loaded dataset with {len(df)} rows')

    df=df[df["UPC"]==target_upc].copy()

    vol_leaked= df.iloc[-1]["units"].values*(1+price_change_pct/100)^elasticity



    return {"target_upc": target_upc,
            "recepients": [recepients]}

if __name__ == "__main__":
    mcp.run(transport="stdio")