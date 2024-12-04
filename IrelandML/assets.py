from dagster import op, job, asset, AssetMaterialization, MetadataValue

import json
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import psycopg2
import matplotlib.pyplot as plt
from io import BytesIO
import base64


@asset(group_name="Amazon_reviews")
def loaded_unstructured_data(): 
    """Loads data from the JSON file into MongoDB."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["unstructured"]
    collection = db["collection"]

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

    collection.insert_many(data)
    cursor = collection.find({})
    data = list(cursor)

    return data  # Return the loaded data

@asset(group_name="Amazon_reviews")
def analyzed_unstructured_data(context, loaded_unstructured_data):
    """Performs data analysis on the loaded data."""
    df = pd.DataFrame(loaded_unstructured_data)

    # --- Create and log plots ---

    # 1. Histogram of 'overall' ratings
    plt.figure()
    plt.hist(df['overall'], bins=5)
    plt.xlabel('Overall Rating')
    plt.ylabel('Frequency')
    plt.title('Distribution of Overall Ratings')

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="analyzed_unstructured_data",
            metadata={
                "overall_ratings_distribution": MetadataValue.md(md_content),
            }
        )
    )

    # 2. Bar chart of 'verified' purchases
    verified_counts = df['verified'].value_counts()
    plt.figure()
    plt.bar(verified_counts.index, verified_counts.values)
    plt.xlabel('Verified Purchase')
    plt.ylabel('Count')
    plt.title('Number of Verified vs. Unverified Purchases')

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="analyzed_unstructured_data",
            metadata={
                "verified_purchases_bar_chart": MetadataValue.md(md_content),
            }
        )
    )
    # --- End of plotting ---

    return df  # Return the analyzed DataFrame

@asset(group_name="Amazon_reviews")
def prediction_unstructured_results(context, analyzed_unstructured_data):
    """
    Performs predictions on the analyzed data and returns 
    prediction results and model metrics.
    """
    features = analyzed_unstructured_data[["overall", "verified", "unixReviewTime"]]
    target = analyzed_unstructured_data["asin"]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # --- Create and log a confusion matrix plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    # ... (add labels and ticks if needed) ...

    # Save and log the plot
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.close()

    context.log_event(
        AssetMaterialization(
            asset_key="prediction_unstructured_results",
            metadata={
                "confusion_matrix_plot": MetadataValue.md(md_content),
            }
        )
    )
    # --- End of plotting ---


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
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


@job
def my_pipeline():
    """Defines the Dagster job."""
    prediction_unstructured_results(analyzed_unstructured_data(loaded_unstructured_data()))