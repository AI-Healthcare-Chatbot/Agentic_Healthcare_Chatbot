from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import json
import re
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import backend functions
from backend.app import clean_plot_references  # Import your clean_plot_references function
from backend.final_agentic_memory import HealthcareDataAgenticAssistant

# Create plots directory if it doesn't exist
PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our assistant
assistant = HealthcareDataAgenticAssistant()

@app.get("/api")
async def root():
    return {"message": "Healthcare Chatbot API is running"}

@app.post("/api/chat")
async def chat(request: Request):
    """REST API endpoint for chat"""
    data = await request.json()
    query = data.get("message", "")
    
    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "No message provided"}
        )
    
    # Add query to chat history
    assistant.chat_history.append({"role": "Human", "message": query})
    
    try:
        # Check if it's general conversation
        if assistant.is_general_conversation(query):
            response = assistant.generate_response_for_general_query(query)
        else:
            # Process through agent
            response = assistant.agent.run(input=query)
            
        # Check if there's a plot in the response
        plot_data = None
        plot_match = re.search(r"(plots[/\\][a-zA-Z0-9_.-]+\.png)", response)
        
        if plot_match:
            # Plot path directly found in the response
            plot_path_str = plot_match.group(1)
            
            # Normalize the path
            normalized_path = plot_path_str.replace('\\', '/')
            
            # Get just the filename
            filename = os.path.basename(normalized_path)
            
            # No need to copy the file - it's already in the plots directory
            plot_url = f"/api/plots/{filename}"
            plot_data = {"url": plot_url}
            
            print(f"Plot detected: {plot_path_str}")
            print(f"Plot URL will be: {plot_url}")
        
        # Clean up the response text to replace file paths with cleaner messages
        clean_response = clean_plot_references(response)
        
        # Add clean response to chat history
        assistant.chat_history.append({"role": "AI", "message": clean_response})
        
        return {
            "response": clean_response,
            "plot": plot_data
        }
        
    except Exception as e:
        print(f"Error processing query: {e}")
        # Use web search as fallback
        try:
            results = assistant.search_handler.search(query)
            fallback_response = assistant.search_handler.format_results(results)
            assistant.chat_history.append({"role": "AI", "message": fallback_response})
            
            return {
                "response": fallback_response
            }
        except Exception as fallback_e:
            error_message = "I'm having trouble processing your request right now. Please try again with a different question."
            assistant.chat_history.append({"role": "AI", "message": error_message})
            
            return JSONResponse(
                status_code=500,
                content={"error": error_message}
            )

@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serve plot images"""
    plot_path = PLOTS_DIR / filename
    if not plot_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Plot not found"}
        )
    return FileResponse(plot_path)

@app.get("/api/history")
async def get_history():
    """Get the conversation history"""
    return {"history": assistant.chat_history}

@app.post("/api/reset")
async def reset_conversation():
    """Reset the conversation history"""
    assistant.chat_history = []
    return {"message": "Conversation memory has been cleared."}