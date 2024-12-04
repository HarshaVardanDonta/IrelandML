import pickle
from dagster import asset, Output, op

@op
def my_op():
    # ... your processing ...
    data_to_materialize = some_python_object
    return data_to_materialize

@asset
def my_asset(my_op):
    data = my_op()
    filepath = r"C:\Users\dhars\OneDrive\Desktop\Rohith\IrelandML\tmphchhn0by\storage\prepared_data"  # Construct the desired filepath

    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

    yield Output(
        value=data, # output the result of my_op for downstream assets if needed
        metadata={"filepath": filepath}, # store the filepath to the pickled data.
    )


