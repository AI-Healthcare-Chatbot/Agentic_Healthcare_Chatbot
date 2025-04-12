import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
import re
from snowflake_connector import SnowflakeConnector

connector = SnowflakeConnector()

SNOWFLAKE_SQLALCHEMY_URI = connector.uri

# Load environment variables
load_dotenv()

# Snowflake connection URI (using SQLAlchemy format)
snowflake_uri = SNOWFLAKE_SQLALCHEMY_URI
db = SQLDatabase.from_uri(snowflake_uri)

# Pinecone setup (modern client)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Create the vector store with MiniLM embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_model,
    text_key="description",  # required key for the content field in your Pinecone vectors
    namespace=None
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Function to clean SQL queries before execution
def clean_sql_query(query):
    # Remove all markdown formatting and backticks
    clean_query = query.replace("```sql", "").replace("```", "").replace("`", "")
    return clean_query.strip()

# Patch the SQLDatabase's run method to clean SQL queries
original_run = SQLDatabase.run

def cleaned_run(self, query):
    clean_query = clean_sql_query(query)
    print("Cleaned SQL before execution:", clean_query)
    return original_run(self, clean_query)

# Apply the patch
SQLDatabase.run = cleaned_run

# SQL tool (LangChain's SQLDatabaseChain)
sql_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True, return_intermediate_steps=True)

# Retriever tool (retrieves metadata or schema context)
retriever_tool = Tool(
    name="SchemaInfoRetriever",
    func=lambda q: retriever.get_relevant_documents(q),
    description="Use this to fetch relevant table or column metadata"
)

# Function to extract and clean SQL query from text
def extract_sql(text):
    # First, remove all markdown formatting and backticks
    text = text.replace("```sql", "").replace("```", "")
    text = text.replace("`", "")
    
    # Try to extract the SQL query by looking for 'SQLQuery:'
    if "SQLQuery:" in text:
        text = text.split("SQLQuery:", 1)[1].strip()
    
    # If there's still no clear SQL query, look for a SELECT statement
    if "SELECT" in text.upper():
        match = re.search(r'(SELECT\s+.*?)(;|\Z)', text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)
            if not text.strip().endswith(";"):
                text = text.strip() + ";"
    
    return text.strip()

# Terminal-based chat loop (for testing without Streamlit)
print("\n💬 Healthcare Data Assistant (Terminal Mode)")
print("Type 'exit' to quit.\n")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Exiting. Have a good day!")
        break
    try:
        # Use retriever and SQL chain independently since SQLDatabaseChain is not a valid agent tool
        print("🔍 Retrieving metadata context...")
        context_docs = retriever.invoke(user_query)
        context_text = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
        
        print("Context from retriever:")
        print(context_text)

        print("🧠 Executing SQL chain...")
        result = sql_chain.invoke({"query": user_query})
        
        # Print raw result for debugging
        print("\nRaw result type:", type(result))
        if isinstance(result, dict):
            print("Result keys:", list(result.keys()))
        
        # Extract the SQL query from intermediate steps
        if isinstance(result, dict) and "intermediate_steps" in result and result["intermediate_steps"]:
            # The SQL is typically in the first intermediate step
            sql_query = str(result["intermediate_steps"][0])
            raw_sql = extract_sql(sql_query)
        else:
            # Fallback to extracting from the final result
            raw_sql = extract_sql(str(result))
        
        print("\n📄 Generated SQL Query:")
        print(raw_sql)
        
        print("\n🧹 Executing query...")
        try:
            # Remove any lingering backticks or formatting before execution
            clean_sql = clean_sql_query(raw_sql)
            # Ensure query starts with SELECT and ends with semicolon
            if not clean_sql.upper().strip().startswith("SELECT"):
                if "SELECT" in clean_sql.upper():
                    clean_sql = clean_sql[clean_sql.upper().find("SELECT"):]
            if not clean_sql.strip().endswith(";"):
                clean_sql = clean_sql.strip() + ";"
                
            print("Final SQL Query to Execute:")
            print(clean_sql)
            
            # The run method has been patched to clean SQL queries
            response = db.run(clean_sql)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"⚠️ SQL Execution Error: {e}")
            print(f"Problematic SQL: {clean_sql}\n")
            
    except Exception as e:
        print(f"⚠️ Error: {e}\n")