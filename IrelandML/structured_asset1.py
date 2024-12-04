from dagster import job, op, asset, ResourceDefinition, Output, AssetMaterialization, MetadataValue
from sqlalchemy import create_engine
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text
from sklearn.preprocessing import LabelEncoder
from .resources import db_connection_resource, create_db_connection
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Asset for raw data import (hardcoded file path)
@asset(resource_defs={"db_connection": db_connection_resource}, group_name="salary_trend")
def raw_data(): 
    """Import raw data from CSV to PostgreSQL."""
    engine = create_db_connection()
    file_path = r"data\structured_data1.csv"  # Hardcoded file path
    df = pd.read_csv(file_path,  on_bad_lines='skip')
    df.to_sql("raw_data", engine, if_exists="replace", index=False)  
    return df


@asset(resource_defs={"db_connection": db_connection_resource}, non_argument_deps={"raw_data"}, group_name="salary_trend")
def analysis_results(context):
    """Perform data analysis and return results."""
    engine = create_db_connection()
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT AVG(Base_Salary) FROM raw_data"))
            avg_value = result.scalar() 

            # Create a histogram
            df = pd.read_sql("SELECT Base_Salary FROM raw_data", engine)
            plt.figure()
            plt.hist(df['Base_Salary'], bins=20)
            plt.xlabel('Base Salary')
            plt.ylabel('Frequency')
            plt.title('Distribution of Base Salaries')

            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()

            # Log the plot as metadata
            context.log_event(
                AssetMaterialization(
                    asset_key="analysis_results",
                    metadata={
                        "avg_salary": avg_value,
                        "salary_distribution": MetadataValue.md(f"<img src='data:image/png;base64,{image_data}'/>"),
                    }
                )
            )

        except SQLAlchemyError as e:
            print(f"Database Error: {e}") 
            avg_value = None 

    

    return {"avg_value": avg_value}




@asset(resource_defs={"db_connection": db_connection_resource}, non_argument_deps={"raw_data"}, group_name="salary_trend")
def prepared_data(context):
    """Prepare data for prediction and return training and testing data."""
    engine = create_db_connection()
    query = "SELECT * FROM raw_data" 
    df = pd.read_sql(query, engine)
    le = LabelEncoder() 

    columns_to_encode = ['Department', 'Department_Name', 'Division', 'Gender', 'Grade']

    for column in columns_to_encode:
        if column in df.columns and df[column].dtype == 'object': 
            try:
                df[column] = le.fit_transform(df[column].astype(str)) 
            except TypeError as e:
                print(f"Error encoding column {column}: {e}")


    target_column = "Base_Salary"
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    y = df[target_column]
    X = df.drop(target_column, axis=1)

    X = X.apply(pd.to_numeric, errors='coerce').dropna()
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test




@asset(group_name="salary_trend")
def prediction_results(prepared_data): 
    X_train, X_test, y_train, y_test = prepared_data
    model = DecisionTreeRegressor(random_state=42) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


@asset(group_name="salary_trend")  
def model_metrics(context, prediction_results, prepared_data): 
    y_pred = prediction_results
    X_train, X_test, y_train, y_test = prepared_data
    mse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))


    # Create a scatter plot
    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Base Salary')
    plt.ylabel('Predicted Base Salary')
    plt.title('Actual vs. Predicted Base Salary')

    # Save the plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.getvalue())
    plt.close()

  

    #     # Create line charts for mse and r2
    # fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # 2 subplots (one for mse, one for r2)

    # # MSE plot
    # axes[0].plot(mse, marker='o', linestyle='-', color='blue')
    # axes[0].set_title('Mean Squared Error (MSE) Over Time')
    # axes[0].set_xlabel('Run')  # Or any relevant x-axis label
    # axes[0].set_ylabel('MSE')

    # # R-squared plot
    # axes[1].plot(r2, marker='o', linestyle='-', color='green')
    # axes[1].set_title('R-squared (R2) Over Time')
    # axes[1].set_xlabel('Run')  # Or any relevant x-axis label
    # axes[1].set_ylabel('R2')

    
    # Convert the image to Markdown to preview it within Dagster
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"
    plt.tight_layout()
    # Log the plot and metrics as metadata
    context.log_event(
        AssetMaterialization(
            asset_key="model_metrics",
            metadata={
                "mse": mse,
                "r2": r2,
                "actual_vs_predicted_plot": MetadataValue.md(md_content),
            }
        )
    )


    return {"mse": mse, "r2": r2, "actual_vs_predicted_plot": image_data}



# Define the pipeline (optional, you can now run assets independently)
@job
def data_pipeline():
    """Pipeline for data import, cleaning, analysis, and prediction."""
    raw_data()
    analysis_results()
    prepared_data()
    prediction_results()
    model_metrics()