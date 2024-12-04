from dagster import job, op, asset, ResourceDefinition, Output, AssetMaterialization, MetadataValue
from sqlalchemy import create_engine
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from .resources import db_connection_resource, create_db_connection
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Asset for raw data import (hardcoded file path)
@asset(resource_defs={"db_connection": db_connection_resource}, group_name="salary_trend")
def raw_data():  # No file_path argument
    """Import raw data from CSV to PostgreSQL."""
    engine = create_db_connection()
    file_path = r"data\structured_data1.csv"  # Hardcoded file path
    df = pd.read_csv(file_path,  on_bad_lines='skip')
    df.to_sql("raw_data", engine, if_exists="replace", index=False)  
    return df




@asset(resource_defs={"db_connection": db_connection_resource}, non_argument_deps={"raw_data"}, group_name="salary_trend")
def analysis_results():
    """Perform data analysis and return results."""
    engine = create_db_connection()
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT AVG(Base_Salary) FROM raw_data")) # Wrap in text()
            avg_value = result.scalar()  # or result.fetchone()[0] if you prefer

        except SQLAlchemyError as e:
            print(f"Database Error: {e}")  # Log the error appropriately
            avg_value = None  # Return None on error. Consider other handling.
            # Consider re-raising the exception if asset failure is desired
            # raise e


    return {"avg_value": avg_value}




@asset(resource_defs={"db_connection": db_connection_resource}, non_argument_deps={"raw_data"}, group_name="salary_trend")  # depends on raw_data
def prepared_data(context):
    """Prepare data for prediction and return training and testing data."""
    engine = create_db_connection()
    # Read data directly from raw_data table
    query = "SELECT * FROM raw_data"  # Now reads from raw_data
    df = pd.read_sql(query, engine)
    # Create LabelEncoder instance
    le = LabelEncoder()  # Create a LabelEncoder object *outside* the loop

    # Columns to encode using label encoding
    columns_to_encode = ['Department', 'Department_Name', 'Division', 'Gender', 'Grade']

    for column in columns_to_encode:
        if column in df.columns and df[column].dtype == 'object':  # Check if column exists and is of type object
            try:
                df[column] = le.fit_transform(df[column].astype(str))  # Apply Label Encoding
            except TypeError as e:
                print(f"Error encoding column {column}: {e}")
                # Handle the error appropriately, e.g., remove the column or impute values


    # Define the target variable
    target_column = "Base_Salary"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    y = df[target_column]
    X = df.drop(target_column, axis=1)


    # Convert all columns in X to numeric if possible.  Replace strings with NaN. Remove rows with NaN values.
    X = X.apply(pd.to_numeric, errors='coerce').dropna()

    # Remove the corresponding y values where X values were removed due to NaN.
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test




@asset(group_name="salary_trend")
def prediction_results(prepared_data):  # Receive the tuple from prepared_data
    X_train, X_test, y_train, y_test = prepared_data
    model = DecisionTreeRegressor(random_state=42)  # Add random_state for reproducibility
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


@asset(group_name="salary_trend")  
def model_metrics(prediction_results, prepared_data): # Get data directly here too.
    y_pred = prediction_results
    X_train, X_test, y_train, y_test = prepared_data
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "r2": r2}



# Define the pipeline (optional, you can now run assets independently)
@job
def data_pipeline():
    """Pipeline for data import, cleaning, analysis, and prediction."""
    raw_data()
    analysis_results()
    prepared_data()
    prediction_results()
    model_metrics()

