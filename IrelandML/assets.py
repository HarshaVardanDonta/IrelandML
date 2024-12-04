from dagster import asset, AssetMaterialization, MetadataValue, job

import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Group name for the assets
GROUP_NAME = "Amazon_Reviews"

@asset(group_name=GROUP_NAME)
def load_unstructured_data():
    """Loads data from the JSON file."""

    with open(r"data\unstructured_data.json", "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = []
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {line} due to error: {e}")

    return data  # Return the loaded data

@asset(group_name=GROUP_NAME)
def analyze_unstructured_data(context, load_unstructured_data):
    """Performs data analysis on the loaded data."""
    df = pd.DataFrame(load_unstructured_data)

    # --- Create and log plots ---

    # 1. Histogram of 'overall' ratings
    plt.figure()
    plt.hist(df["overall"], bins=5)
    plt.xlabel("Overall Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Overall Ratings")

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="analyze_unstructured_data",
            metadata={
                "overall_ratings_distribution": MetadataValue.md(md_content),
            },
        )
    )

    # 2. Bar chart of 'verified' purchases
    verified_counts = df["verified"].value_counts()
    plt.figure()
    plt.bar(verified_counts.index, verified_counts.values)
    plt.xlabel("Verified Purchase")
    plt.ylabel("Count")
    plt.title("Number of Verified vs. Unverified Purchases")

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="analyze_unstructured_data",
            metadata={
                "verified_purchases_bar_chart": MetadataValue.md(md_content),
            },
        )
    )
    # --- End of plotting ---

    return df  # Return the analyzed DataFrame

@asset(group_name=GROUP_NAME)
def preprocess_unstructured_data(context, analyze_unstructured_data):
    """Preprocesses the analyzed data."""
    df = analyze_unstructured_data.copy()

    # Convert 'verified' to numeric
    df["verified"] = df["verified"].astype(int)

    # Scale numeric features
    scaler = StandardScaler()
    df[["overall", "unixReviewTime"]] = scaler.fit_transform(
        df[["overall", "unixReviewTime"]]
    )

    return df  # Return the preprocessed DataFrame

@asset(group_name=GROUP_NAME)
def predict_unstructured_results(context, preprocess_unstructured_data):
    """
    Performs predictions on the analyzed data and returns
    prediction results and model metrics.
    """
    df = preprocess_unstructured_data.copy()
    features = df[["overall", "verified", "unixReviewTime"]]
    target = df["asin"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # --- Create and log a confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    # ... (add labels and ticks if needed) ...

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="predict_unstructured_results",
            metadata={
                "confusion_matrix_plot": MetadataValue.md(md_content),
            },
        )
    )

    # --- Create and log a prediction plot ---

    # Convert y_test and y_pred to numeric for plotting
    y_test_numeric = pd.factorize(y_test)[0]
    y_pred_numeric = pd.factorize(y_pred)[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_numeric, y_pred_numeric, alpha=0.5)
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Actual vs. Predicted Values", fontsize=14)
    plt.grid(True)

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="predict_unstructured_results",
            metadata={
                "prediction_plot": MetadataValue.md(md_content),
            },
        )
    )

    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # Return predictions and metrics
    return {
        "predictions": y_pred.tolist(),  # Convert to list for serialization
        "accuracy": accuracy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

@job
def my_unstructured_pipeline():
    """Defines the Dagster job."""
    predict_unstructured_results(
        preprocess_unstructured_data(
            analyze_unstructured_data(load_unstructured_data())
        )
    )