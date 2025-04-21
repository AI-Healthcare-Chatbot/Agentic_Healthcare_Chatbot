import requests
import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

class WebSearchHandler:
    """
    Handles web search functionality through Tavily API for queries 
    that cannot be answered from the database.
    """
    
    def __init__(self):
        """Initialize the web search handler with API credentials."""
        # Load environment variables if not already loaded
        load_dotenv()
        
        # Initialize Tavily API key from environment variables
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        if not self.tavily_api_key:
            print("Warning: Tavily API key not found. Web search will not function.")
            print("Please sign up at https://tavily.com and add TAVILY_API_KEY to your .env file.")
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Perform a web search using Tavily API.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            
        Returns:
            Dict containing search results and metadata
        """
        if not self.tavily_api_key:
            return {
                "success": False,
                "error": "Tavily API key not configured",
                "results": []
            }
            
        try:
            # Enhance query with healthcare context if needed
            search_query = self._enhance_query(query)
            
            # Call Tavily API
            url = "https://api.tavily.com/search"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "api_key": self.tavily_api_key,
                "query": search_query,
                "search_depth": "advanced",
                "include_domains": [
                    "mayoclinic.org", "nih.gov", "cdc.gov", "who.int", 
                    "webmd.com", "medlineplus.gov", "healthline.com", 
                    "hopkinsmedicine.org", "clevelandclinic.org"
                ],
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False
            }
            
            response = requests.post(url, headers=headers, data=json.dumps(data))
            result = response.json()
            
            # Process and format the results
            if "results" in result:
                return {
                    "success": True,
                    "query": search_query,
                    "answer": result.get("answer", ""),
                    "results": result["results"]
                }
            else:
                return {
                    "success": False,
                    "error": "No results found",
                    "query": search_query,
                    "results": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def _enhance_query(self, query: str) -> str:
        """
        Enhance the query with healthcare context if needed.
        
        Args:
            query: The original query
            
        Returns:
            Enhanced query with healthcare context
        """
        # Check if query already has healthcare context
        healthcare_terms = ["healthcare", "medical", "patient", "hospital", 
                           "clinical", "disease", "treatment", "doctor", 
                           "nurse", "diagnosis", "health"]
        
        has_healthcare_context = any(term in query.lower() for term in healthcare_terms)
        
        # Add healthcare context if needed
        if not has_healthcare_context:
            return f"{query} healthcare medical information"
        
        return query
    
    def format_results(self, search_results: Dict[str, Any]) -> str:
        """
        Format search results into natural language.
        
        Args:
            search_results: The results from the search method
            
        Returns:
            A formatted string with search results in natural language
        """
        if not search_results["success"]:
            return f"I couldn't find information on that topic in our database, and the web search also encountered an issue: {search_results.get('error', 'Unknown error')}."
        
        if not search_results["results"]:
            return "I couldn't find relevant information on that topic in our database or from reliable healthcare sources online."
        
        # If Tavily provided a direct answer, use it
        if search_results.get("answer") and len(search_results["answer"]) > 50:
            formatted = search_results["answer"]
            
            # Add sources information
            formatted += "\n\nThis information is based on the following sources:\n"
            for i, result in enumerate(search_results["results"][:3], 1):
                formatted += f"{i}. {result.get('title', 'Unnamed source')} - {result.get('url', 'No URL available')}\n"
                
            return formatted
        
        # If no direct answer, create our own summary
        formatted = f"I couldn't find this information in our healthcare database, but here's what I found from reliable medical sources:\n\n"
        
        # Add each result with source attribution
        for i, result in enumerate(search_results["results"], 1):
            title = result.get("title", "Unnamed source")
            snippet = result.get("content", "No content available")
            source = result.get("url", "No URL available")
            
            formatted += f"**From {title}**:\n{snippet}\n"
            formatted += f"Source: {source}\n\n"
        
        formatted += "\nPlease note that this information comes from external medical sources rather than our internal database."
        
        return formatted    