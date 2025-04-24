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
        
        # ─── Chat History ─────────────────────────────────────────────
        self.chat_history = []
        
        # ─── Initialize Tools and Prompts ─────────────────────────────
        self.setup_tools_and_prompts()
    
    def generate_clean_title(self, query: str, plot_type: str) -> str:
        """Generate a clean, readable title for plots"""
        # Extract key elements from the query
        query = query.lower()
        
        # Find what we're showing
        subject_keywords = {
            "hospital": "Hospitals",
            "diagnos": "Diagnoses", 
            "disease": "Diagnoses",
            "procedure": "Procedures",
            "age": "Age Groups",
            "region": "Regions",
            "race": "Demographics",
            "ethnic": "Demographics"
        }
        
        subject = "Data"  # Default
        for keyword, label in subject_keywords.items():
            if keyword in query:
                subject = label
                break
        
        # Find the metric
        metric_keywords = {
            "cost": "Costs",
            "charge": "Costs",
            "expense": "Costs",
            "count": "Count",
            "stay": "Length of Stay"
        }
        
        metric = "Distribution"  # Default
        for keyword, label in metric_keywords.items():
            if keyword in query:
                metric = label
                break
        
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
        title = f"{plot_type.capitalize()} Chart: "
        if limiter:
            title += f"{limiter} {subject} by {metric}"
        else:
            title += f"{subject} by {metric}"
        
        return title
        
    def setup_tools_and_prompts(self):
        # ─── Create SQL generation prompt ───
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
        
        # ─── Create Response Formatter ───
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
        
        # ─── Define Agent Tools ───────────────────────────────────────
        
        # Schema info tool
        def get_schema_info(query: str) -> str:
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
        
        # SQL Generation and execution function
        def generate_and_run_sql(query: str) -> List[Tuple[Any, ...]]:
            print(f"\n💻 Generating SQL for: {query}")
            try:
                # Get formatted chat history
                formatted_history = self.format_chat_history()
                
                # Get schema info
                full_schema_info = self.db.get_table_info()
                schema_only = re.sub(r'/\*.*?\*/', '', full_schema_info, flags=re.DOTALL).strip()
                
                # Get metadata from retriever
                print("🔍 Retrieving metadata for SQL generation...")
                context_docs = self.retriever.invoke(query)
                metadata_text = "\n".join([doc.page_content for doc in context_docs if doc.page_content])
                
                # Try broader search if needed
                if not metadata_text or len(metadata_text) < 100:
                    broader_terms = []
                    
                    # Map keywords to search terms
                    keyword_mapping = {
                        "age": ["age", "age_group"],
                        "serious": ["severity", "apr_severity_of_illness_description"],
                        "severe": ["severity", "apr_severity_of_illness_description"],
                        "diagnosis": ["diagnosis", "ccsr_diagnosis_description"],
                        "disease": ["diagnosis", "ccsr_diagnosis_description"],
                        "hospital": ["hospital", "facility_name"]
                    }
                    
                    # Add relevant broader terms
                    for keyword, terms in keyword_mapping.items():
                        if keyword in query.lower():
                            broader_terms.extend(terms)
                    
                    if broader_terms:
                        print(f"🔍 Using broader terms for metadata: {', '.join(broader_terms)}")
                        for term in broader_terms:
                            additional_docs = self.retriever.invoke(term)
                            additional_text = "\n".join([doc.page_content for doc in additional_docs if doc.page_content])
                            metadata_text += "\n" + additional_text
                
                print(f"Found metadata ({len(metadata_text)} chars)")
                
                # Combine schema and metadata
                combined_info = f"""
                DATABASE SCHEMA:
                {schema_only}
                
                COLUMN METADATA:
                {metadata_text}
                """
                
                print("🧠 Using non-agentic approach for SQL generation...")
                
                # Check if this is a follow-up query about diseases
                is_followup_for_diseases = (
                    any(word in query.lower() for word in ["disease", "diagnos"]) and 
                    any(word in query.lower() for word in ["these", "those", "them"]) and
                    any(entry["message"].lower().find("70 or older") != -1 
                        for entry in self.chat_history if entry["role"] == "Human")
                )
                
                # Special handling for follow-up queries
                if is_followup_for_diseases:
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
                    print(f"✅ Using specialized SQL for top diseases in elderly serious patients")
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
                        common_fixes = {
                            r'\bFROM patients\b': "FROM hospital_data",
                            r'\bseverity_of_illness\b': "apr_severity_of_illness_description"
                        }
                        
                        for pattern, replacement in common_fixes.items():
                            if re.search(pattern, clean_sql):
                                clean_sql = re.sub(pattern, replacement, clean_sql)
                                print(f"⚠️ Fixed SQL: {pattern} -> {replacement}")
                    
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
        
        # Plot generator tool
        def generate_plot(query: str) -> str:
            print(f"\n📈 Generating plot for: {query}")
            try:
                # Get the last SQL results
                if not hasattr(self, '_last_results') or not self._last_results:
                    return "❌ No SQL results available to plot. Please run a query first."
                
                print(f"Data for plotting: {self._last_results[:5]} (showing first 5 rows)")
                
                # Determine plot type from query
                plot_type = "bar"  # Default
                if any(term in query.lower() for term in ["pie", "distribution", "breakdown"]):
                    plot_type = "pie"
                elif any(term in query.lower() for term in ["bar", "top", "most"]):
                    plot_type = "bar"
                    
                # Generate a clean title
                clean_title = self.generate_clean_title(query, plot_type)
                plot_query = f"{plot_type} chart of {query}"
                
                # Try primary plot type
                plot_result = self.plot_generator.process_query_result(
                    {"result": f"Data from query: {self._last_sql}", "title": clean_title}, 
                    plot_query, 
                    self._last_results
                )
                
                # If first attempt fails, try alternative plot type
                if not plot_result or plot_result.startswith("Unable to extract data"):
                    alt_plot_type = "bar" if plot_type == "pie" else "pie"
                    alt_plot_query = f"{alt_plot_type} chart of {self._last_query}"
                    print(f"First attempt failed. Trying alternative plot type: {alt_plot_query}")
                    
                    plot_result = self.plot_generator.process_query_result(
                        {"result": f"Data from query: {self._last_sql}", "title": clean_title}, 
                        alt_plot_query, 
                        self._last_results
                    )
                
                if plot_result and not plot_result.startswith("Unable to extract data"):
                    print(f"✅ Plot generated successfully")
                    filename = os.path.basename(plot_result)
                    return f"✅ Plot generated successfully: /plots/{filename}"
                
                print("❌ Plot generation failed")
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
        
        # Response formatter
        def format_response(query: str) -> str:
            print(f"\n📝 Formatting response for: {query}")
            try:
                # Check if we have results to format
                if not hasattr(self, '_last_results'):
                    return "No query results available to format."
                
                # Use the LLM for formatting
                try:
                    formatted_response = self.response_formatter.invoke({
                        "question": query or self._last_query,
                        "sql_query": getattr(self, '_last_sql', "Unknown SQL query"),
                        "db_result": str(self._last_results)
                    })
                    
                    print("✅ Response formatted successfully by LLM")
                    return formatted_response.get("text", "")
                except Exception as llm_error:
                    print(f"⚠️ Error using LLM for formatting: {llm_error}")
                    
                    # Basic fallback formatting
                    if not self._last_results:
                        return "No results were found for your query."
                        
                    # Handle single value results
                    if len(self._last_results) == 1 and isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 1:
                        value = self._last_results[0][0]
                        is_cost = any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"])
                        
                        if isinstance(value, (int, float)):
                            return f"The {'value is ${:,.2f}' if is_cost else 'result is {:,}'}.".format(value)
                    
                    # Handle multi-row results
                    elif self._last_results:
                        # For single column results
                        if isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 1:
                            items = [str(row[0]) for row in self._last_results[:10]]
                            return f"Results: {', '.join(items)}"
                        
                        # For name-value pairs (common in distribution queries)
                        elif isinstance(self._last_results[0], tuple) and len(self._last_results[0]) == 2:
                            is_cost = any(term in query.lower() for term in ["cost", "charge", "expense", "price", "expensive"])
                            
                            response = "Results:\n\n"
                            for i, (name, value) in enumerate(self._last_results[:10], 1):
                                if isinstance(value, (int, float)):
                                    value_str = "${:,.2f}".format(value) if is_cost else "{:,}".format(value)
                                    response += f"{i}. {name}: {value_str}\n"
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
You are a healthcare data assistant with access to New York State Patient Discharge Database for 2022 and metadata about the database schema.
When answering questions about healthcare data, follow these steps in order:

1. Call GetSchemaInfo to understand what data is available in the database
2. Call RetrieveMetadata to get detailed information about relevant columns
3. Call GenerateAndRunSQL to query the database
4. Call FormatResponse to create a user-friendly response
5. If the user specifically asked for a visualization or the data is suitable for plotting, call GeneratePlot
6. Only if steps 1-4 fail to provide adequate information, call WebSearch as a last resort

 

When answering questions about what you can do or what all you can be asked, always mention:
1. You have access to the complete 2022 New York State Patient Discharge Database
2. You can analyze hospital stays, diagnoses, procedures, and costs specifically for New York hospitals in 2022
3. You can generate visualizations like bar charts and pie charts of the New York healthcare data
4. You can compare different hospitals, regions, or patient demographics within New York
5. You can provide insights on medical conditions, but your primary strength is analyzing the NY discharge data
6. You can also search the web for additional information if needed

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
        
    # ─── Helper Methods ─────────────────────────────────────────
    def clean_sql_query(self, query):
        return query.replace("```sql", "").replace("```", "").replace("`", "").strip()

    def extract_sql(self, text):
        """Extract SQL query from text"""
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
        """Detect general conversation patterns"""
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
        """Format chat history for prompts"""
        formatted_history = ""
        for entry in self.chat_history:
            formatted_history += f"{entry['role']}: {entry['message']}\n"
        return formatted_history
        
    def generate_response_for_general_query(self, query):
        """Handle general conversation queries"""
        formatted_history = self.format_chat_history()
        
        prompt = f"""
        You are a knowledgeable healthcare data assistant with access to the 2022 New York State Patient Discharge Database. You can answer
        questions about healthcare data, procedures, diagnoses, and medical information.
        
        When asked what you can do, always mention that your primary function is to analyze the New York State 
    Patient Discharge Database from 2022, and that users can ask about hospitals, diagnoses, procedures, 
    patient demographics, costs, and request visualizations of this specific dataset.
    
        
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
            
            # Handle special commands
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Exiting. Have a good day!")
                break
            elif user_query.lower() == "clear memory":
                self.memory.clear()
                self.chat_history = []
                print("🧹 Conversation memory has been cleared.")
                continue
            elif user_query.lower() == "tokens":
                tokens_available = self.rate_limiter.tokens
                cooldown_time = 0 if tokens_available >= 3 else (3 - tokens_available) / self.rate_limiter.fill_rate
                print(f"🔢 Available API tokens: {tokens_available:.1f}/{self.rate_limiter.capacity}")
                print(f"⏱️ Time until next query possible: {max(0, cooldown_time):.1f} seconds")
                continue
                
            # Add query to history
            self.chat_history.append({"role": "Human", "message": user_query})
                
            # Handle general conversation
            if self.is_general_conversation(user_query):
                print("\U0001F4AC Processing conversational query...")
                try:
                    response = self.generate_response_for_general_query(user_query)
                    print(f"\nAssistant: {response}\n")
                    self.chat_history.append({"role": "AI", "message": response})
                    continue
                except Exception as e:
                    print(f"⚠️ Error handling general conversation: {e}")
                    # Fall through to agent processing
            
            # Process through the agent
            try:
                print("\n⚙️ Processing your query (this may take a moment)...")
                start_time = time.time()
                response = self.agent.run(input=user_query)
                process_time = time.time() - start_time
                print(f"✅ Query processed in {process_time:.1f} seconds")
                print(f"\nAssistant: {response}\n")
                self.chat_history.append({"role": "AI", "message": response})
                
            except Exception as e:
                print(f"\n⚠️ Error processing query: {e}")
                print("I encountered an error while processing your request. Let me search the web for information.")
                
                # Fallback to web search
                try:
                    results = self.search_handler.search(user_query)
                    fallback_response = self.search_handler.format_results(results)
                    print(f"\nAssistant: {fallback_response}\n")
                    self.chat_history.append({"role": "AI", "message": fallback_response})
                except Exception as fallback_e:
                    error_message = "I'm having trouble processing your request right now. Please try again with a different question or try again later."
                    print(f"\nAssistant: {error_message}\n")
                    self.chat_history.append({"role": "AI", "message": error_message})


if __name__ == "__main__":
    assistant = HealthcareDataAgenticAssistant()
    assistant.run()