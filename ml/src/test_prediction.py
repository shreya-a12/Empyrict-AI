from data_preprocessing import preprocess_data
from prediction_pipeline import (
    predict_single_employee,
    predict_bulk_employees
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    label_encoder
) = preprocess_data(verbose=False)


# ============================================================
# SINGLE EMPLOYEE TEST
# ============================================================

sample_employee = X_test.iloc[[0]]

actual_label = label_encoder.inverse_transform(
    [y_test[0]]
)[0]

prediction = predict_single_employee(sample_employee)

print("=" * 60)
print("SINGLE EMPLOYEE ATTRITION PREDICTION")
print("=" * 60)

print(f"Actual Label            : {actual_label}")
print(f"Predicted Label         : {prediction['Prediction']}")
print(f"Probability of Leaving  : {prediction['Probability_Leaving']}%")
print(f"Probability of Staying  : {prediction['Probability_Staying']}%")
print(f"Risk Level              : {prediction['Risk_Level']}")
print(f"Recommendation          : {prediction['Recommendation']}")


# ============================================================
# BULK EMPLOYEE TEST
# ============================================================

print("\n")
print("=" * 60)
print("BULK EMPLOYEE ATTRITION PREDICTION")
print("=" * 60)

sample_employees = X_test.iloc[:5]

bulk_predictions = predict_bulk_employees(sample_employees)

actual_labels = label_encoder.inverse_transform(
    y_test[:5]
)

bulk_predictions["Actual_Label"] = actual_labels

print(
    bulk_predictions[
        [
            "Actual_Label",
            "Prediction",
            "Probability_Leaving",
            "Probability_Staying",
            "Risk_Level"
        ]
    ]
)

print("\nPrediction Successful!")