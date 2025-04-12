from snowflake_connector import SnowflakeConnector
from sqlalchemy import text

# Initialize the connector
connector = SnowflakeConnector()

# Get the engine and run a query
# engine = connector.get_engine()
# with engine.connect() as conn:
#     result = conn.execute(text("SELECT CURRENT_USER(), CURRENT_REGION(), CURRENT_TIMESTAMP()"))
#     for row in result:
#         print(row)

print(connector.uri)