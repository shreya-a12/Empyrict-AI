import os
import json
import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

current_dir = os.path.dirname(__file__)

models_path = os.path.join(current_dir, "..", "models")

deployment_config_path = os.path.join(
    models_path,
    "deployment_config.json"
)

feature_columns_path = os.path.join(
    models_path,
    "feature_columns.pkl"
)

scaler_path = os.path.join(
    models_path,
    "scaler.pkl"
)


# ============================================================
# LOAD DEPLOYMENT CONFIGURATION
# ============================================================

if not os.path.exists(deployment_config_path):
    raise FileNotFoundError(
        "deployment_config.json not found."
    )

with open(deployment_config_path, "r") as file:
    deployment_config = json.load(file)

model_file = deployment_config["model_file"]
requires_scaling = deployment_config["requires_scaling"]


# ============================================================
# LOAD MODEL
# ============================================================

model_path = os.path.join(
    models_path,
    model_file
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"{model_file} not found."
    )

model = joblib.load(model_path)


# ============================================================
# LOAD FEATURE COLUMNS
# ============================================================

if not os.path.exists(feature_columns_path):
    raise FileNotFoundError(
        "feature_columns.pkl not found."
    )

feature_columns = joblib.load(
    feature_columns_path
)


# ============================================================
# LOAD SCALER (ONLY IF REQUIRED)
# ============================================================

scaler = None

if requires_scaling:

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "scaler.pkl not found."
        )

    scaler = joblib.load(scaler_path)


# ============================================================
# PREPROCESS INPUT DATA
# ============================================================

def preprocess_input(df):

    df = df.copy()

    # Drop unnecessary columns if present
    columns_to_drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # One Hot Encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )

    # Add missing columns
    for column in feature_columns:

        if column not in df.columns:
            df[column] = 0

    # Remove extra columns
    extra_columns = [
        column
        for column in df.columns
        if column not in feature_columns
    ]

    if extra_columns:
        df.drop(
            columns=extra_columns,
            inplace=True
        )

    # Arrange columns
    df = df[feature_columns]

    # Scaling
    if requires_scaling:
        df = scaler.transform(df)

    return df


# ============================================================
# RISK LEVEL
# ============================================================

def classify_risk(probability):

    if probability < 30:
        return "Low"

    elif probability < 70:
        return "Medium"

    return "High"


# ============================================================
# HR RECOMMENDATION
# ============================================================

def generate_recommendation(probability):

    if probability < 30:

        return (
            "Employee has a low attrition risk. "
            "Continue regular engagement."
        )

    elif probability < 70:

        return (
            "Employee has a moderate attrition risk. "
            "Schedule periodic check-ins."
        )

    return (
        "Employee has a high attrition risk. "
        "Immediate HR intervention is recommended."
    )
# ============================================================
# SINGLE EMPLOYEE PREDICTION
# ============================================================

def predict_single_employee(df):

    """
    Predict attrition for a single employee.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing exactly one employee.

    Returns
    -------
    dict
        Prediction result.
    """

    processed_data = preprocess_input(df)

    prediction = model.predict(processed_data)[0]
    probabilities = model.predict_proba(processed_data)[0]

    probability_leaving = round(probabilities[1] * 100, 2)
    probability_staying = round(probabilities[0] * 100, 2)

    result = {
        "Prediction": "Yes" if prediction == 1 else "No",
        "Probability_Leaving": probability_leaving,
        "Probability_Staying": probability_staying,
        "Risk_Level": classify_risk(probability_leaving),
        "Recommendation": generate_recommendation(
            probability_leaving
        )
    }

    return result


# ============================================================
# BULK EMPLOYEE PREDICTION
# ============================================================

def predict_bulk_employees(df):

    """
    Predict attrition for multiple employees.

    Parameters
    ----------
    df : pandas.DataFrame
        Employee dataset.

    Returns
    -------
    pandas.DataFrame
        Original data with prediction columns.
    """

    processed_data = preprocess_input(df)

    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)

    probability_leaving = probabilities[:, 1] * 100
    probability_staying = probabilities[:, 0] * 100

    result = df.copy()

    result["Prediction"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    result["Probability_Leaving"] = (
        probability_leaving.round(2)
    )

    result["Probability_Staying"] = (
        probability_staying.round(2)
    )

    result["Risk_Level"] = [
        classify_risk(probability)
        for probability in probability_leaving
    ]

    result["Recommendation"] = [
        generate_recommendation(probability)
        for probability in probability_leaving
    ]

    return result


# ============================================================
# MAIN (TESTING)
# ============================================================

if __name__ == "__main__":

    dataset_path = os.path.join(
        current_dir,
        "..",
        "..",
        "dataset",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Dataset not found."
        )

    dataset = pd.read_csv(dataset_path)

    sample_employee = dataset.drop(
        columns=["Attrition"]
    ).iloc[[0]]

    print("\nSingle Employee Prediction")
    print("=" * 60)

    single_prediction = predict_single_employee(
        sample_employee
    )

    for key, value in single_prediction.items():
        print(f"{key}: {value}")

    print("\nBulk Employee Prediction")
    print("=" * 60)

    bulk_prediction = predict_bulk_employees(
        sample_employee
    )

    print(bulk_prediction)