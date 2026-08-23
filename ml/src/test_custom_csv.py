import os
import pandas as pd

from prediction_pipeline import predict_bulk_employees


# ============================================================
# PATH
# ============================================================

current_dir = os.path.dirname(__file__)

csv_path = os.path.join(
    current_dir,
    "..",
    "..",
    "dataset",
    "test_employees.csv"
)


# ============================================================
# LOAD CSV
# ============================================================

employees = pd.read_csv(csv_path)

print("=" * 60)
print("INPUT EMPLOYEES")
print("=" * 60)
print(employees)


# ============================================================
# PREDICT
# ============================================================

predictions = predict_bulk_employees(employees)


print("\n")
print("=" * 60)
print("PREDICTION RESULTS")
print("=" * 60)

print(
    predictions[
        [
            "Prediction",
            "Probability_Leaving",
            "Probability_Staying",
            "Risk_Level",
            "Recommendation"
        ]
    ]
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output_path = os.path.join(
    current_dir,
    "..",
    "..",
    "dataset",
    "prediction_results.csv"
)

predictions.to_csv(
    output_path,
    index=False
)

print("\nPrediction file saved successfully!")
print(output_path)