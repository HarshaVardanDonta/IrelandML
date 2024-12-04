from dagster import job, op, asset, ResourceDefinition
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



@asset(resource_defs={"db_connection": db_connection_resource}, group_name="mortality")
def raw_data_2():
    """Import raw data from structured_data2.csv to PostgreSQL."""
    engine = create_db_connection()
    file_path = r"data\structured_data2.csv"  # Path to the second CSV file
    df = pd.read_csv(file_path, on_bad_lines='skip')
    df.to_sql("raw_data_2", engine, if_exists="replace", index=False)  # Use a different table name
    return df



@asset(resource_defs={"db_connection": db_connection_resource}, non_argument_deps={"raw_data_2"}, group_name="mortality")  # depends on raw_data_2
def prepared_data_2(): # Exactly the same as prepared_data but references the new table 'raw_data_2' and modified to take account of the differences in the CSV data.
    """Prepare data for prediction (structured_data2.csv) and return training and testing data."""
    engine = create_db_connection()
    query = "SELECT * FROM raw_data_2"  # Query the new table
    df = pd.read_sql(query, engine)

     # Create LabelEncoder instance
    le = LabelEncoder()  # Create a LabelEncoder object *outside* the loop

    # Columns to encode using label encoding
    columns_to_encode = ['Year', 'Race', 'Sex', 'Age-adjusted Death Rate']
    
    # Convert string columns and ensure target variable is float: modify if different columns in structured_data2.csv
    # Use same label encoder as before.

    for column in columns_to_encode:
        if column in df.columns and df[column].dtype == 'object':
            try:
                df[column] = le.fit_transform(df[column].astype(str))
            except TypeError as e:
                print(f"Error encoding column {column}: {e}")

    # Define the target variable
    target_column = "Average Life Expectancy (Years)"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Convert to numeric and drop NaN rows in X *BEFORE* dropping NaN rows from y related to target_column
    X = df.drop(columns=[target_column])
    X = X.apply(pd.to_numeric, errors='coerce')  # Replace invalid numeric values with NaN immediately.
    X = X.dropna()

    # Now handle missing values in the target column. It is important to drop NaNs in 'y' *after* modifying X to handle NaNs.
    y = df.loc[X.index, target_column].dropna()  # Drop NaNs in 'y' based on the index after NaN handling for X
    X = X[X.index.isin(y.index)] 

    y = df[target_column]
    X = df.drop(target_column, axis=1)


    # Convert all columns in X to numeric if possible.  Replace strings with NaN. Remove rows with NaN values.
    X = X.apply(pd.to_numeric, errors='coerce').dropna()

    # Remove the corresponding y values where X values were removed due to NaN.
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    return X_train, X_test, y_train, y_test



@asset(group_name="mortality")
def prediction_results_2(prepared_data_2): # depends on prepared_data_2
    X_train, X_test, y_train, y_test = prepared_data_2
    model = DecisionTreeRegressor(random_state=42)  # Add random_state for reproducibility
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred




@asset(group_name="mortality")
def model_metrics_2(prediction_results_2, prepared_data_2):
    y_pred = prediction_results_2
    X_train, X_test, y_train, y_test = prepared_data_2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "r2": r2}





@job
def data_pipeline(): # structured_assets, unstructured_assets must be called here
    """Pipeline for data import, cleaning, analysis, and prediction."""
    raw_data_2()
    prepared_data_2()
    prediction_results_2()
    model_metrics_2()

