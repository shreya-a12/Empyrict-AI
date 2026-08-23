import os
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data_preprocessing import preprocess_data

# Load preprocessed data
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

# Models directory
current_dir = os.path.dirname(__file__)
models_path = os.path.join(current_dir, "..", "models")

# Load models
logistic_model = joblib.load(
    os.path.join(models_path, "logistic_regression.pkl")
)

random_forest_model = joblib.load(
    os.path.join(models_path, "random_forest.pkl")
)


def evaluate_model(name, model, X):

    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        roc_auc = "N/A"

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    if roc_auc != "N/A":
        print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }


# Evaluate Logistic Regression
logistic_results = evaluate_model(
    "Logistic Regression",
    logistic_model,
    X_test_scaled
)

# Evaluate Random Forest
random_forest_results = evaluate_model(
    "Random Forest",
    random_forest_model,
    X_test
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"{'Metric':<15}{'Logistic':<15}{'Random Forest'}")
print("-" * 60)

metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

for metric in metrics:

    lr = logistic_results[metric]
    rf = random_forest_results[metric]

    if isinstance(lr, float):
        lr = f"{lr:.4f}"

    if isinstance(rf, float):
        rf = f"{rf:.4f}"

    print(f"{metric:<15}{lr:<15}{rf}")

print("\n" + "=" * 60)

best_model_path = os.path.join(models_path, "best_model.pkl")

# Select Best Model
if logistic_results["F1"] >= random_forest_results["F1"]:
    best_model = logistic_model
    best_model_name = "Logistic Regression"
    model_file = "logistic_regression.pkl"
    requires_scaling = True
else:
    best_model = random_forest_model
    best_model_name = "Random Forest"
    model_file = "random_forest.pkl"
    requires_scaling = False

# Save Best Model
joblib.dump(best_model, best_model_path)

# Save Model Metrics
metrics = {
    "best_model": best_model_name,
    "logistic_regression": {
        "accuracy": logistic_results["Accuracy"],
        "precision": logistic_results["Precision"],
        "recall": logistic_results["Recall"],
        "f1_score": logistic_results["F1"],
        "roc_auc": logistic_results["ROC_AUC"]
    },
    "random_forest": {
        "accuracy": random_forest_results["Accuracy"],
        "precision": random_forest_results["Precision"],
        "recall": random_forest_results["Recall"],
        "f1_score": random_forest_results["F1"],
        "roc_auc": random_forest_results["ROC_AUC"]
    }
}

metrics_path = os.path.join(models_path, "model_metrics.json")

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

# Save Deployment Configuration
deployment_config = {
    "best_model": best_model_name,
    "model_file": model_file,
    "requires_scaling": requires_scaling
}

deployment_config_path = os.path.join(
    models_path,
    "deployment_config.json"
)

with open(deployment_config_path, "w") as file:
    json.dump(deployment_config, file, indent=4)

print(f"Best Model: {best_model_name}")
print("Best model saved successfully!")
print("Model metrics saved successfully!")
print("Deployment configuration saved successfully!")

print("=" * 60)