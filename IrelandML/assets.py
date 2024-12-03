from dagster import op, job, asset

import json
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import psycopg2


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
def analyzed_unstructured_data(loaded_unstructured_data):
    """Performs data analysis on the loaded data."""
    df = pd.DataFrame(loaded_unstructured_data)
    print(df.head())
    print(df.describe())
    print(df.info())

    return df  # Return the analyzed DataFrame

@asset(group_name="Amazon_reviews")
def prediction_unstructured_results(analyzed_unstructured_data):
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