from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from snowflake_connector import SnowflakeConnector

load_dotenv()
connector = SnowflakeConnector()
# snowflake_uri = os.getenv("SNOWFLAKE_SQLALCHEMY_URI")

# print(snowflake_uri)  # Should show %40 and not decode it back to @

snowflake_uri = connector.uri

engine = create_engine(snowflake_uri)

with engine.connect() as conn:
    result = conn.execute(text("SELECT CURRENT_ACCOUNT(), CURRENT_REGION(), CURRENT_TIMESTAMP()"))
    for row in result:
        print(row)