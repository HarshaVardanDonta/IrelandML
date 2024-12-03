# resources.py
from dagster import ResourceDefinition, op
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:Iamdiablo@localhost:5432/postgres"  # Or your actual URL

@op
def create_db_connection():  # No arguments needed
    engine = create_engine(DATABASE_URL)
    return engine


db_connection_resource = ResourceDefinition.hardcoded_resource(create_db_connection)

