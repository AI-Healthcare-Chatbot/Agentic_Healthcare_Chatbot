# api/index.py
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your healthcare assistant
from backend.final_agentic_memory import HealthcareDataAgenticAssistant

# Initialize the assistant
assistant = HealthcareDataAgenticAssistant()

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "Healthcare Chatbot API is running"
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
        
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        # Process based on the path
        if self.path == '/api/chat':
            # Handle chat request
            query = data.get('message', '')
            
            # Add to chat history
            assistant.chat_history.append({"role": "Human", "message": query})
            
            # Generate response
            response = assistant.generate_response_for_general_query(query)
            
            # Add to chat history
            assistant.chat_history.append({"role": "AI", "message": response})
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            self.wfile.write(json.dumps({
                "response": response,
                "plot": None  # Would need additional logic for plots
            }).encode())
            
        elif self.path == '/api/reset':
            # Reset conversation
            assistant.chat_history = []
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            self.wfile.write(json.dumps({
                "message": "Conversation memory has been cleared."
            }).encode())
            
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            self.wfile.write(json.dumps({
                "error": "Endpoint not found"
            }).encode())