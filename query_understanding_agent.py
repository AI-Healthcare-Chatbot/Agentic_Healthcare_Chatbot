import os
import json
import logging
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryUnderstandingAgent:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize QueryUnderstandingAgent with OpenAI model

        Args:
            model (str): OpenAI model to use for query understanding
        """
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.model = model

    def understand_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Analyze and understand the user's healthcare data query

        Args:
            query (str): Natural language query from user

        Returns:
            Optional[Dict[str, Any]]: Structured query understanding
        """
        system_prompt = """
        You are an expert healthcare data query analyzer. 
        For each query, provide a comprehensive breakdown:
        - Precise intent of the query
        - Key medical/statistical concepts
        - Type of information requested
        - Potential relevant data columns
        """

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this query: {query}"}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            logger.error(f"Query understanding error: {e}")
            return None

    def extract_keywords(self, query: str) -> list:
        """
        Extract key medical and statistical keywords from query

        Args:
            query (str): User's query

        Returns:
            list: List of extracted keywords
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract important keywords from the query"},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=100
            )

            keywords = response.choices[0].message.content.split(',')
            return [keyword.strip() for keyword in keywords]

        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
            return []

# Example usage
def main():
    agent = QueryUnderstandingAgent()
    query = "What are the total medical costs for patients with respiratory diseases?"
    
    understanding = agent.understand_query(query)
    keywords = agent.extract_keywords(query)
    
    print("Query Understanding:", understanding)
    print("Keywords:", keywords)

if __name__ == "__main__":
    main()