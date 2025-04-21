import os
import re
import time
import subprocess
from typing import Any, Dict, List, Tuple, Optional
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings

from pinecone import Pinecone
from langchain_pinecone import Pinecone as PineconeVectorStore

from snowflake_connector import SnowflakeConnector
from web_search_handler import WebSearchHandler
from gen_color_plot import GeneratingPlots

class TokenBucket:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, tokens, fill_rate):
        """
        Initialize token bucket
        tokens - initial number of tokens
        fill_rate - tokens added per second
        """
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_time = time.time()
        
    def consume(self, tokens=1):
        """
        Consume tokens from the bucket
        Returns wait time needed if not enough tokens
        """
        # Refill tokens based on time elapsed
        now = time.time()
        time_passed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + time_passed * self.fill_rate)
        self.last_time = now
        
        # Check if we have enough tokens
        if tokens <= self.tokens:
            self.tokens -= tokens
            return 0  # No wait time needed
        else:
            # Calculate wait time to have enough tokens
            additional_tokens_needed = tokens - self.tokens
            wait_time = additional_tokens_needed / self.fill_rate
            return wait_time

class HealthcareDataAgenticAssistant:
    def __init__(self):
        load_dotenv()
        
        # ─── Initialize Components ────────────────────────────────────
        self.connector = SnowflakeConnector()
        self.plot_generator = GeneratingPlots()
        self.search_handler = WebSearchHandler()
        self.db = SQLDatabase.from_uri(self.connector.uri)
        
        # ─── Pinecone Vector Store Setup ─────────────────────────────
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        
        # Setup vector store retriever
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=embedding_model,
            text_key="combined_text",
            namespace=None
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # ─── Rate Limiter for API Calls ───────────────────────────────
        # Allow 15 API calls per minute (0.25 per second)
        self.rate_limiter = TokenBucket(tokens=15, fill_rate=0.25)
        
        # ─── LLM (Gemini) ─────────────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        
        # ─── Simple Memory Implementation (like non-agentic version) ───
        self.chat_history = []
        
        # ─── Initialize Tools and Prompts ─────────────────────────────
        self.setup_tools_and_prompts()
    
    def generate_clean_title(self, query: str, plot_type: str) -> str:
        """Generate a clean, readable title for plots"""
        # Extract key elements from the query
        query = query.lower()
        
        # Find what we're showing
        subject = ""
        if "hospital" in query:
            subject = "Hospitals"
        elif "diagnos" in query or "disease" in query:
            subject = "Diagnoses"
        elif "procedure" in query:
            subject = "Procedures"
        elif "age" in query:
            subject = "Age Groups"
        elif "region" in query:
            subject = "Regions"
        elif "race" in query or "ethnic" in query:
            subject = "Demographics"
        else:
            # Use a generic title if no specific subject found
            subject = "Data"
        
        # Find the metric
        metric = ""
        if "cost" in query or "charge" in query or "expense" in query:
            metric = "Costs"
        elif "count" in query:
            metric = "Count"
        elif "stay" in query:
            metric = "Length of Stay"
        else:
            metric = "Distribution"
        
        # Find any limiters
        limiter = ""
        if "top" in query:
            # Find the number after "top"
            match = re.search(r"top\s+(\d+)", query)
            if match:
                limiter = f"Top {match.group(1)}"
            else:
                limiter = "Top"
        
        # Build the title
        if limiter:
            title = f"{plot_type.capitalize()} Chart: {limiter} {subject} by {metric}"
        else:
            title = f"{plot_type.capitalize()} Chart: {subject} by {metric}"
        
        return title
        
    def setup_tools_and_prompts(self):
        # ─── Create SQL generation prompt (similar to non-agentic version) ───
        self.query_generator_prompt = PromptTemplate.from_template("""
        You are a smart data assistant with access to a healthcare database.
        
        Previous conversation:
        {chat_history}
        
        Current user question: {user_question}
        
        Database schema information:
        {table_info}
        
        Based on the conversation history and the current question, generate a SQL query that will answer the user's question.
        When generating SQL queries:
        - If filtering on text fields (e.g., diagnosis, procedure description), avoid using '='.
        - Instead, break the important words into individual terms and use ILIKE with OR logic:
            Example: For "heart surgery", use:
            ccsr_procedure_description ILIKE '%heart%' OR ccsr_procedure_description ILIKE '%surgery%'
        - Consider the conversation history for context when generating queries.
        - If the question seems to be a follow-up to a previous query, maintain context appropriately.
        - IMPORTANT: Focus ONLY on the current question. Do not be influenced by previous unrelated queries.
        - If the query mentions terms like "expensive", "cost", "charge", etc., make sure to use appropriate aggregation.
        - If the query asks for "top N" or "most expensive", include ORDER BY and LIMIT clauses.
        - Make sure to include COUNT(*) when needed for aggregation, especially for plots or distributions.
        - When asking for "top" diseases, diagnoses, etc., always include the COUNT in the SELECT clause.
        
        Return ONLY the SQL query, nothing else.
        """)
        
        self.query_generator = LLMChain(
            llm=self.llm,
            prompt=self.query_generator_prompt,
            verbose=False
        )
        
        # ─── Create Response Formatter (similar to non-agentic version) ───
        self.response_formatter_prompt = PromptTemplate.from_template("""
        You are a helpful healthcare data assistant.
        
        Original user question: {question}
        SQL query that was executed: {sql_query}
        Raw database result: {db_result}
        
        Format this database result into a natural, helpful response for the user.
        - Use proper sentences and formatting
        - Include the actual values from the results
        - Format currency values with dollar signs and commas as appropriate
        - Provide context about what the numbers represent
        - Be concise but informative
        - If the result is about costs or charges, make it clear these are averages from the database
        
        Your response:
        """)
        
        self.response_formatter = LLMChain(
            llm=self.llm,
            prompt=self.response_formatter_prompt,
            verbose=False
        )
        
        # ─── Agent Memory ───────────────────────────────────────────
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )
        
        # ─── Create Agent Tools ───────────────────────────────────────
        # Schema info tool - modified to exclude example rows
        def get_schema_info(query: str) -> str:
            """Only called directly by the agent, not by other functions"""
            print(f"\n🔍 Getting schema info for: {query}")
            try:
                full_schema = self.db.get_table_info()
                # Remove the example rows section
                schema_only = re.sub(r'/\*.*?\*/', '', full_schema, flags=re.DOTALL)
                print(f"✅ Retrieved schema info successfully")
                return schema_only.strip()
            except Exception as e:
                print(f"❌ Error getting schema info: {e}")
                return f"Error retrieving schema info: {str(e)}"
        
        # Metadata retrieval tool
        def retrieve_metadata(query: str) -> str:
            """Only called directly by the agent, not by other functions"""
            print(f"\n🔍 Retrieving metadata for: {query}")
            try:
                context_docs = self.retriever.invoke(query)
                if not context_docs:
                    print("⚠️ No metadata found")
                    return "No relevant metadata found."
                
                context_text = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
                print(f"✅ Retrieved metadata: {len(context_docs)} documents")
                return context_text
            except Exception as e:
                print(f"❌ Error retrieving metadata: {e}")
                return f"Error retrieving metadata: {str(e)}"
                
        # SQL Generation function using non-agentic approach
        def generate_and_run_sql(query: str) -> List[Tuple[Any, ...]]:
            print(f"\n💻 Generating SQL for: {query}")
            try:
                # Get formatted chat history from our non-agentic approach
                formatted_history = self.format_chat_history()
                
                # Get full schema info without the example rows
                full_schema_info = self.db.get_table_info()
                schema_only = re.sub(r'/\*.*?\*/', '', full_schema_info, flags=re.DOTALL).strip()
                
                # Get metadata from retriever
                print("🔍 Retrieving metadata for SQL generation...")
                context_docs = self.retriever.invoke(query)
                metadata_text = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
                
                if not metadata_text or len(metadata_text) < 100:
                    # Try a broader query if specific query yields limited metadata
                    broader_terms = []
                    
                    if "age" in query.lower():
                        broader_terms.append("age")
                        broader_terms.append("age_group")
                        
                    if "serious" in query.lower() or "severe" in query.lower():
                        broader_terms.append("severity")
                        broader_terms.append("apr_severity_of_illness_description")
                        
                    if "diagnosis" in query.lower() or "disease" in query.lower():
                        broader_terms.append("diagnosis")
                        broader_terms.append("ccsr_diagnosis_description")
                        
                    if "hospital" in query.lower():
                        broader_terms.append("hospital")
                        broader_terms.append("facility_name")
                    
                    if broader_terms:
                        print(f"🔍 Using broader terms for metadata: {', '.join(broader_terms)}")
                        for term in broader_terms:
                            additional_docs = self.retriever.invoke(term)
                            additional_text = "\n".join([doc.page_content for doc in additional_docs if doc.page_content])
                            metadata_text += "\n" + additional_text
                
                # Print the metadata being used
                print(f"Found metadata ({len(metadata_text)} chars)")
                
                # Combine schema and metadata
                combined_info = f"""
                DATABASE SCHEMA:
                {schema_only}
                
                COLUMN METADATA:
                {metadata_text}
                """
                
                # Generate the SQL query using the non-agentic approach
                print("🧠 Using non-agentic approach for SQL generation...")
                
                # Check if this is a follow-up query about diseases in a specific group
                is_followup_for_diseases = any(word in query.lower() for word in ["disease", "diagnos"]) and any(word in query.lower() for word in ["these", "those", "them"])
                
                # Special handling for follow-up queries about top diseases in a specific group
                if is_followup_for_diseases and any(entry["message"].lower().find("70 or older") != -1 for entry in self.chat_history if entry["role"] == "Human"):
                    clean_sql = """
                    SELECT ccsr_diagnosis_description, COUNT(*) as count 
                    FROM hospital_data 
                    WHERE age_group = '70 or Older' AND 
                          (apr_severity_of_illness_description = 'Major' OR 
                           apr_severity_of_illness_description = 'Extreme')
                    GROUP BY ccsr_diagnosis_description
                    ORDER BY count DESC
                    LIMIT 5;
                    """
                    print(f"✅ Using specialized SQL for top diseases in elderly serious patients: {clean_sql}")
                else:
                    try:
                        contextualized_query = self.query_generator.invoke({
                            "chat_history": formatted_history,
                            "user_question": query,
                            "table_info": combined_info
                        })
                        
                        # Extract the SQL query
                        clean_sql = self.extract_sql(contextualized_query.get("text", ""))
                        print(f"✅ Generated SQL: {clean_sql}")
                        
                        # Fix common issues with the SQL query
                        if "FROM patients" in clean_sql:
                            clean_sql = clean_sql.replace("FROM patients", "FROM hospital_data")
                            print(f"⚠️ Fixed table name in SQL: {clean_sql}")
                            
                        # Use regex with word boundaries to avoid partial matches
                        if re.search(r'\bseverity_of_illness\b', clean_sql):
                            clean_sql = re.sub(r'\bseverity_of_illness\b', "apr_severity_of_illness_description", clean_sql)
                            print(f"⚠️ Fixed severity column name in SQL: {clean_sql}")
                    
                    except Exception as gen_error:
                        print(f"⚠️ Error in SQL generation: {gen_error}")
                        raise gen_error
                
                # Execute the SQL query
                print(f"📊 Executing SQL...")
                results = self.db.run(clean_sql)
                print(f"✅ SQL execution complete: {len(results) if results else 0} rows")
                
                # Save the results for later use
                self._last_query = query
                self._last_sql = clean_sql
                self._last_results = results
                
                return results
                    
            except Exception as e:
                print(f"❌ Error generating or running SQL: {e}")
                return []
                
        # Plot generator tool with improved handling for pie charts
        def generate_plot(query: str) -> str:
            print(f"\n📈 Generating plot for: {query}")
            try:
                # Get the last SQL results
                if not hasattr(self, '_last_results') or not self._last_results:
                    return "❌ No SQL results available to plot. Please run a query first."
                
                print(f"Data for plotting: {self._last_results[:5]} (showing first 5 rows)")
                
                # Determine the appropriate plot type based on the query and data
                plot_type = "bar"
                
                # Explicit pie chart request should take priority
                if "pie" in query.lower() or "distribution" in query.lower() or "breakdown" in query.lower():
                    plot_type = "pie"
                # Only set to bar if pie wasn't explicitly requested
                elif "bar" in query.lower() or "top" in query.lower() or "most" in query.lower():
                    plot_type = "bar"
                    
                # Generate a clean title
                clean_title = self.generate_clean_title(query, plot_type)
                
                # Construct a plot-friendly query
                plot_query = f"{plot_type} chart of {query}"
                print(f"Using plot query: {plot_query}")
                
                # Generate the plot with a custom title
                plot_result = self.plot_generator.process_query_result(
                    {"result": f"Data from query: {self._last_sql}", "title": clean_title}, 
                    plot_query, 
                    self._last_results
                )
                
                if plot_result and not plot_result.startswith("Unable to extract data"):
                    print(f"✅ Plot generated successfully: {plot_result}")
                    
                    # Try to display the plot
                    try:
                        if os.path.exists(plot_result):
                            subprocess.Popen(['open', plot_result], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    except Exception as e:
                        print(f"⚠️ Could not open plot file: {e}")
                    
                    return f"✅ Plot generated successfully: {plot_result}"
                else:
                    # Try the alternative plot type if first attempt failed
                    alt_plot_type = "bar" if plot_type == "pie" else "pie"
                    alt_plot_query = f"{alt_plot_type} chart of {self._last_query}"
                    print(f"First attempt failed. Trying alternative plot type: {alt_plot_query}")
                    
                    alt_plot_result = self.plot_generator.process_query_result(
                        {"result": f"Data from query: {self._last_sql}", "title": clean_title}, 
                        alt_plot_query, 
                        self._last_results
                    )
                    
                    if alt_plot_result and not alt_plot_result.startswith("Unable to extract data"):
                        print(f"✅ Alternative plot generated successfully: {alt_plot_result}")
                        
                        # Try to display the plot
                        # try:
                        #     if os.path.exists(alt_plot_result):
                        #         subprocess.Popen(['open', alt_plot_result], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                        # except Exception as e:
                        #     print(f"⚠️ Could not open plot file: {e}")
                        
                        filename = os.path.basename(alt_plot_result)
                        return f"✅ Plot generated successfully: /plots/{filename}"

                        # return f"✅ Plot generated successfully: {alt_plot_result}"
                    
                    print("❌ Both plot attempts failed")
                    return "❌ Unable to generate a meaningful plot from this data. The data might not be suitable for visualization."
            except Exception as e:
                print(f"❌ Error generating plot: {e}")
                return f"Error generating plot: {str(e)}"
                
        # Web search fallback
        def web_search(query: str) -> str:
            print(f"\n🌐 Searching web for: {query}")
            try:
                results = self.search_handler.search(query)
                formatted_results = self.search_handler.format_results(results)
                print("✅ Web search complete")
                return formatted_results
            except Exception as e:
                print(f"❌ Error performing web search: {e}")
                return f"Error performing web search: {str(e)}"
                
        # Response formatter using non-agentic approach
        def format_response(query: str) -> str:
            print(f"\n📝 Formatting response for: {query}")
            try:
                # Check if we have results to format
                if not hasattr(self, '_last_results'):
                    return "No query results available to format."
                
                # Always try to use the LLM for final formatting first
                try:
                    formatted_response = self.response_formatter.invoke({
                        "question": query or self._last_query,
                        "sql_query": getattr(self, '_last_sql', "Unknown SQL query"),
                        "db_result": str(self._last_results)
                    })
                    
                    print("✅ Response formatted successfully by LLM")
                    return formatted_response
                except Exception as llm_error:
                    print(f"⚠️ Error using LLM for formatting: {llm_error}")
                    
                    # Manual formatting as fallback
                    if isinstance(self._last_results, list) and len(self._last_results) == 1 and isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 1:
                        value = self._last_results[0][0]
                        if isinstance(value, (int, float)):
                            if any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"]):
                                return f"The value is ${value:,.2f}."
                            else:
                                return f"The result is {value:,}."
                    
                    # Format multi-row results
                    elif isinstance(self._last_results, list) and len(self._last_results) > 0:
                        # Determine if we have a single column or multiple columns
                        if isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 1:
                            return f"Results: {', '.join(str(row[0]) for row in self._last_results[:10])}"
                        
                        # For name-value pairs (common in distribution queries)
                        elif isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 2:
                            response = "Results:\n\n"
                            for i, (name, value) in enumerate(self._last_results[:10], 1):
                                if isinstance(value, (int, float)):
                                    if any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"]):
                                        response += f"{i}. {name}: ${value:,.2f}\n"
                                    else:
                                        response += f"{i}. {name}: {value:,}\n"
                                else:
                                    response += f"{i}. {name}: {value}\n"
                            return response
                    
                    # Generic fallback
                    return f"Query results: {str(self._last_results)}"
            except Exception as e:
                print(f"❌ Error formatting response: {e}")
                return f"Error formatting response: {str(e)}"
                
        # Define the tools
        self.tools = [
            Tool("GetSchemaInfo", get_schema_info, "Get database schema information"),
            Tool("RetrieveMetadata", retrieve_metadata, "Retrieve metadata about database columns"),
            Tool("GenerateAndRunSQL", generate_and_run_sql, "Generate and execute SQL query"),
            Tool("GeneratePlot", generate_plot, "Generate a plot from query results"),
            Tool("WebSearch", web_search, "Search the web for information"),
            Tool("FormatResponse", format_response, "Format query results into a user-friendly response")
        ]
        
        # ─── Agent Prompting ──────────────────────────────────────────
        prefix = """
You are a healthcare data assistant with access to a medical database and metadata about the database schema.
When answering questions about healthcare data, follow these steps in order:

1. Call GetSchemaInfo to understand what data is available in the database
2. Call RetrieveMetadata to get detailed information about relevant columns
3. Call GenerateAndRunSQL to query the database
4. Call FormatResponse to create a user-friendly response
5. If the user specifically asked for a visualization or the data is suitable for plotting, call GeneratePlot
6. Only if steps 1-4 fail to provide adequate information, call WebSearch as a last resort

Important guidelines:
- Focus on the current question only, don't get distracted by previous context
- If the question is about "top" or "most expensive", make sure to use ORDER BY and LIMIT
- If the question is about costs or charges, use appropriate aggregation functions
- Handle each query independently unless it's clearly a follow-up question
- For "serious" or "severe" conditions, look for metadata on severity classification

For general questions or conversation not requiring database access, respond directly.
"""

        suffix = """
Chat history:
{chat_history}

Now, answer the following new question:
{input}
"""

        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent="zero-shot-react-description",
            handle_parsing_errors=True,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "prefix": prefix,
                "suffix": suffix,
                "input_variables": ["chat_history", "input"],
            },
        )
        
    # ─── Helper Methods from Non-Agentic Version ─────────────────────
    def clean_sql_query(self, query):
        return query.replace("```sql", "").replace("```", "").replace("`", "").strip()

    def extract_sql(self, text):
        """Extract SQL query from text using the same approach as non-agentic version"""
        text = text.replace("```sql", "").replace("```", "").replace("`", "")
        if "SQLQuery:" in text:
            text = text.split("SQLQuery:", 1)[1].strip()
        if "SELECT" in text.upper():
            match = re.search(r'(SELECT\s+.*?)(;|\Z)', text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)
                if not text.strip().endswith(";"):
                    text = text.strip() + ";"
        return text.strip()
        
    def is_general_conversation(self, query):
        """Detect general conversation patterns from non-agentic version"""
        conversation_patterns = [
            r'^(hi|hello|hey|greetings|good (morning|afternoon|evening))',
            r'^how (are|is|do) you',
            r'^(thanks|thank you)',
            r'^(what can you do|help me|what are your capabilities)',
            r'^\?+$',
            r'^(ok|okay|got it|i see|understood)',
        ]
        return any(re.match(pattern, query.lower()) for pattern in conversation_patterns)
        
    def format_chat_history(self):
        """Format chat history similar to non-agentic version"""
        formatted_history = ""
        for entry in self.chat_history:
            formatted_history += f"{entry['role']}: {entry['message']}\n"
        return formatted_history
        
    def generate_response_for_general_query(self, query):
        """Handle general conversation similar to non-agentic version"""
        formatted_history = self.format_chat_history()
        
        prompt = f"""
        You are a knowledgeable healthcare data assistant. You have access to medical databases and can answer
        questions about healthcare data, procedures, diagnoses, and medical information.
        
        Previous conversation:
        {formatted_history}
        
        Human: {query}
        AI Assistant:
        """
        
        response = self.llm.predict(prompt)
        return response
    
    # ─── Main Run Method ─────────────────────────────────────────────
    def run(self):
        print("\n🏥 Agentic Healthcare Data Assistant (Terminal Mode)")
        print("Type 'exit' to quit.")
        print("Type 'clear memory' to reset conversation history.")
        print("Type 'tokens' to check remaining API tokens.")
        print("\n⚙️ Rate limit enabled: 15 API calls per minute to prevent quota exhaustion\n")

        while True:
            # Check rate limit before accepting a new query
            # Each query costs approximately 3 tokens (API calls)
            wait_time = self.rate_limiter.consume(3)
            if wait_time > 0:
                print(f"\n⏳ API rate limit reached. Cooling down for {wait_time:.1f} seconds...")
                # Show a countdown timer
                start_time = time.time()
                elapsed = 0
                while elapsed < wait_time:
                    elapsed = time.time() - start_time
                    remaining = max(0, wait_time - elapsed)
                    print(f"\rCooldown: {remaining:.1f} seconds remaining...   ", end="", flush=True)
                    time.sleep(0.5)
                print("\rReady for your next query!                          ")
            
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Exiting. Have a good day!")
                break
            elif user_query.lower() == "clear memory":
                self.memory.clear()
                self.chat_history = []  # Clear both memory types
                print("🧹 Conversation memory has been cleared.")
                continue
            elif user_query.lower() == "tokens":
                tokens_available = self.rate_limiter.tokens
                cooldown_time = 0 if tokens_available >= 3 else (3 - tokens_available) / self.rate_limiter.fill_rate
                print(f"🔢 Available API tokens: {tokens_available:.1f}/{self.rate_limiter.capacity}")
                print(f"⏱️ Time until next query possible: {max(0, cooldown_time):.1f} seconds")
                continue
                
            # Check if it's general conversation
            if self.is_general_conversation(user_query):
                print("\U0001F4AC Processing conversational query...")
                try:
                    response = self.generate_response_for_general_query(user_query)
                    print(f"\nAssistant: {response}\n")
                    
                    # Add to both memory systems
                    self.chat_history.append({"role": "Human", "message": user_query})
                    self.chat_history.append({"role": "AI", "message": response})
                    continue
                except Exception as e:
                    print(f"⚠️ Error handling general conversation: {e}")
                    # Fall through to agent processing
            
            # Add to history for agent processing
            self.chat_history.append({"role": "Human", "message": user_query})
                
            try:
                # Process the query through the agent with retries
                max_retries = 1  # Just try once to avoid quota issues
                for attempt in range(max_retries):
                    try:
                        # Display a processing message
                        print("\n⚙️ Processing your query (this may take a moment)...")
                        
                        # Get start time for tracking processing duration
                        start_time = time.time()
                        
                        # Process the query
                        response = self.agent.run(input=user_query)
                        
                        # Calculate and display processing time
                        process_time = time.time() - start_time
                        print(f"✅ Query processed in {process_time:.1f} seconds")
                        
                        print(f"\nAssistant: {response}\n")
                        
                        # Add to non-agentic history
                        self.chat_history.append({"role": "AI", "message": response})
                        break
                    except Exception as e:
                        # Just let it fall through to web search
                        raise e
            except Exception as e:
                print(f"\n⚠️ Error processing query: {e}")
                print("I encountered an error while processing your request. Let me search the web for information.")
                
                # Fallback to web search as last resort
                try:
                    results = self.search_handler.search(user_query)
                    fallback_response = self.search_handler.format_results(results)
                    print(f"\nAssistant: {fallback_response}\n")
                    
                    # Add to chat history
                    self.chat_history.append({"role": "AI", "message": fallback_response})
                except Exception as fallback_e:
                    print(f"\nAssistant: I'm having trouble processing your request right now. Please try again with a different question or try again later.\n")
                    
                    # Add failure to chat history
                    self.chat_history.append({
                        "role": "AI", 
                        "message": "I'm having trouble processing your request right now. Please try again with a different question or try again later."
                    })


if __name__ == "__main__":
    assistant = HealthcareDataAgenticAssistant()
    assistant.run()