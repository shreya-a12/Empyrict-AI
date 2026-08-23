import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(verbose=True):

    # Get the current file's directory
    current_dir = os.path.dirname(__file__)

    # Build the path to the dataset
    dataset_path = os.path.join(
        current_dir,
        "..",
        "..",
        "dataset",
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    # Normalize the path
    dataset_path = os.path.abspath(dataset_path)

    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Display first 5 rows
    if verbose:
        print("\nFirst 5 Rows:")
        print(df.head())

    # Dataset shape
    if verbose:
        print("\nDataset Shape:")
        print(df.shape)

    # Display dataset information
    if verbose:
        print("\nDataset Information:")
        df.info()

    # Check for missing values
    if verbose:
        print("\nMissing Values:")
        print(df.isnull().sum())

    # Display statistical summary of numerical columns
    if verbose:
        print("\nStatistical Summary:")
        print(df.describe())

    # Display all column names
    if verbose:
        print("\nColumn Names:")
        print(df.columns.tolist())

    # Display target variable distribution
    if verbose:
        print("\nAttrition Value Counts:")
        print(df["Attrition"].value_counts())

    # Display target variable percentages
    if verbose:
        print("\nAttrition Percentage:")
        print(df["Attrition"].value_counts(normalize=True) * 100)

    # Check duplicate rows
    if verbose:
        print("\nDuplicate Rows:")
        print(df.duplicated().sum())

    # Check columns with only one unique value
    if verbose:
        print("\nColumns with only one unique value:")
        for column in df.columns:
            if df[column].nunique() == 1:
                print(column)

    # Remove unnecessary columns
    df = df.drop(
        columns=[
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ]
    )

    # Separate features and target
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    if verbose:
        print("\nFeature Matrix Shape:", X.shape)
        print("Target Vector Shape:", y.shape)

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if verbose:
        print("\nEncoded Target Values:")
        print(y[:10])

    # One-Hot Encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    if verbose:
        print("\nFeature Matrix After Encoding:")
        print(X.shape)

    # Create models directory if it doesn't exist
    models_path = os.path.join(current_dir, "..", "models")
    os.makedirs(models_path, exist_ok=True)

    # Save feature column names
    feature_columns_path = os.path.join(models_path, "feature_columns.pkl")
    joblib.dump(X.columns.tolist(), feature_columns_path)

    if verbose:
        print("\nFeature columns saved successfully!")

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if verbose:
        print("\nTraining Set:", X_train.shape)
        print("Testing Set:", X_test.shape)

    # Scale the data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if verbose:
        print("\nData Scaling Completed.")

    # Save scaler
    scaler_path = os.path.join(models_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    if verbose:
        print("\nScaler saved successfully!")

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoder
    )


if __name__ == "__main__":
    preprocess_data()