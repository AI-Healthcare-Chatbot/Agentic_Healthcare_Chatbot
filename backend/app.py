import os
import json
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Any, Optional
import base64
from pathlib import Path
import tempfile
import shutil
import re

# Import our healthcare data assistant
from final_agentic_memory_Cleaned import HealthcareDataAgenticAssistant

app = FastAPI(title="Healthcare Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a plots directory if it doesn't exist
PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Serve plot images statically
app.mount("/plots", StaticFiles(directory="plots"), name="plots")

# Initialize our assistant
assistant = HealthcareDataAgenticAssistant()
print("✅ Healthcare assistant initialized!")

# Store active connections
active_connections: Dict[str, WebSocket] = {}

# Function to clean up plot references in the text
def clean_plot_references(text):
    # Handle the specific format showing in your interface
    text = re.sub(r"Here is the plot for the top 5 diagnoses for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", 
                 "Here is the plot for the top 5 diagnoses for patients over 70:", text)
    
    # More general patterns to catch /plots/ path formats
    text = re.sub(r"for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", "for patients over 70:", text)
    text = re.sub(r"for the top 5 diagnoses for patients over 70: /plots/[a-zA-Z0-9_.-]+\.png", 
                 "for the top 5 diagnoses for patients over 70:", text)
    
    # Handle other common patterns
    text = re.sub(r"The plot is saved as plots\\[a-zA-Z0-9_.-]+\.png\.?", "Here's the visualization:", text)
    text = re.sub(r"The plot is saved as plots/[a-zA-Z0-9_.-]+\.png\.?", "Here's the visualization:", text)
    text = re.sub(r"The plot is saved as .+\.png\.?", "Here's the visualization:", text)
    
    # Replace direct references to plot paths
    text = re.sub(r"Here is the bar plot for the top 5 diseases: plots\\[a-zA-Z0-9_.-]+\.png", "Here is the bar plot for the top 5 diseases:", text)
    text = re.sub(r"Here is the bar plot of the top 5 diseases: plots\\[a-zA-Z0-9_.-]+\.png", "Here is the bar plot of the top 5 diseases:", text)
    
    # Generic replacements
    text = re.sub(r": plots\\[a-zA-Z0-9_.-]+\.png", ":", text)
    text = re.sub(r": plots/[a-zA-Z0-9_.-]+\.png", ":", text)
    text = re.sub(r": /plots/[a-zA-Z0-9_.-]+\.png", ":", text)
    
    # Special case for the exact pattern in the screenshot
    text = re.sub(r"Here is the plot for the top 5 diagnoses for patients over 70: /plots/plot_[a-zA-Z0-9_.-]+\.png", 
                 "Here is the plot for the top 5 diagnoses for patients over 70:", text)
    
    # Clean up any double spaces or awkward punctuation
    text = re.sub(r"\s\s+", " ", text)
    text = re.sub(r"\.\s+\.", ".", text)
    
    return text.strip()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
            
    async def send_plot(self, plot_path: str, client_id: str):
        if client_id in self.active_connections:
            # Extract filename from path, normalize slashes
            filename = os.path.basename(plot_path.replace('\\', '/'))
            # Convert local path to URL
            plot_url = f"/plots/{filename}"
            plot_data = {
                "type": "plot",
                "url": plot_url
            }
            await self.active_connections[client_id].send_text(json.dumps(plot_data))

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Healthcare Chatbot API is running"}

# Extra route to check if a plot exists
@app.get("/check-plot/{filename}")
async def check_plot(filename: str):
    plot_path = PLOTS_DIR / filename
    if plot_path.exists():
        return {"exists": True}
    return {"exists": False}

# Direct access to a plot file
@app.get("/plot/{filename}")
async def get_plot(filename: str):
    plot_path = PLOTS_DIR / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(plot_path)

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    print(f"Client {client_id} connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            query = message.get("message", "")
            reset = message.get("reset", False)
            
            if reset:
                # Reset the assistant's memory
                assistant.chat_history = []
                await manager.send_message(json.dumps({
                    "type": "response",
                    "message": "Conversation memory has been cleared."
                }), client_id)
                continue
            
            if not query:
                await manager.send_message(json.dumps({
                    "type": "error",
                    "message": "No message provided"
                }), client_id)
                continue
            
            # Send "thinking" message
            await manager.send_message(json.dumps({
                "type": "thinking",
                "message": "Processing your query..."
            }), client_id)
            
            # Process the query
            try:
                # Add query to chat history
                assistant.chat_history.append({"role": "Human", "message": query})
                
                # Check if it's general conversation
                if assistant.is_general_conversation(query):
                    response = assistant.generate_response_for_general_query(query)
                    assistant.chat_history.append({"role": "AI", "message": response})
                    
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": response
                    }), client_id)
                else:
                    # Process through agent
                    response = await asyncio.to_thread(assistant.agent.run, input=query)
                    
                    # Check if there's a plot in the response using improved pattern
                    plot_path = None
                    
                    # Improved pattern to catch all plot path formats
                    plot_match = re.search(r"(/plots/[a-zA-Z0-9_.-]+\.png|plots[/\\][a-zA-Z0-9_.-]+\.png)", response)
                    
                    if plot_match:
                        # Plot path directly found in the response
                        plot_path_str = plot_match.group(1)
                        
                        # Normalize the path 
                        normalized_path = plot_path_str.replace('\\', '/')
                        
                        # Get just the filename
                        filename = os.path.basename(normalized_path)
                        
                        # The file is already in the plots directory - no need to copy!
                        plot_path = str(Path("plots") / filename)
                        
                        print(f"Plot detected: {plot_path}")
                        print(f"Plot URL will be: /plots/{filename}")
                    
                    # Clean up the response text to replace file paths with cleaner messages
                    clean_response = clean_plot_references(response)
                    
                    # Add clean response to chat history
                    assistant.chat_history.append({"role": "AI", "message": clean_response})
                    
                    # Send the text response
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": clean_response
                    }), client_id)
                    
                    # If there was a plot, send it separately
                    if plot_path:
                        # Add a slight delay to ensure the text response is processed first
                        await asyncio.sleep(0.5)
                        await manager.send_plot(plot_path, client_id)
                    else:
                        print("No plot found in response")
            
            except Exception as e:
                print(f"Error processing query: {e}")
                # Use web search as fallback
                try:
                    results = assistant.search_handler.search(query)
                    fallback_response = assistant.search_handler.format_results(results)
                    assistant.chat_history.append({"role": "AI", "message": fallback_response})
                    
                    await manager.send_message(json.dumps({
                        "type": "response",
                        "message": fallback_response
                    }), client_id)
                except Exception as fallback_e:
                    error_message = "I'm having trouble processing your request right now. Please try again with a different question."
                    assistant.chat_history.append({"role": "AI", "message": error_message})
                    
                    await manager.send_message(json.dumps({
                        "type": "error",
                        "message": error_message
                    }), client_id)
                    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        print(f"Client {client_id} disconnected")

@app.post("/api/chat")
async def chat(request: Request):
    """REST API endpoint for chat (alternative to WebSocket)"""
    data = await request.json()
    query = data.get("message", "")
    
    if not query:
        raise HTTPException(status_code=400, detail="No message provided")
    
    # Add query to chat history
    assistant.chat_history.append({"role": "Human", "message": query})
    
    try:
        # Check if it's general conversation
        if assistant.is_general_conversation(query):
            response = assistant.generate_response_for_general_query(query)
        else:
            # Process through agent
            response = assistant.agent.run(input=query)
            
        # Check if there's a plot in the response with improved pattern
        plot_data = None
        plot_match = re.search(r"(/plots/[a-zA-Z0-9_.-]+\.png|plots[/\\][a-zA-Z0-9_.-]+\.png)", response)
        
        if plot_match:
            # Plot path directly found in the response
            plot_path_str = plot_match.group(1)
            
            # Normalize the path
            normalized_path = plot_path_str.replace('\\', '/')
            
            # Get just the filename
            filename = os.path.basename(normalized_path)
            
            # No need to copy the file - it's already in the plots directory
            plot_url = f"/plots/{filename}"
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

@app.get("/api/history")
async def get_history():
    """Get the conversation history"""
    return {"history": assistant.chat_history}

@app.post("/api/reset")
async def reset_conversation():
    """Reset the conversation history"""
    assistant.chat_history = []
    return {"message": "Conversation memory has been cleared."}

if __name__ == "__main__":
    # Use the PORT environment variable if available (for cloud deployment)
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)