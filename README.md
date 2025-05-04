# Agentic Healthcare Data Assistant

![Healthcare Data Assistant Banner](https://img.shields.io/badge/Healthcare-AI%20Assistant-4caf50)
![Agentic AI](https://img.shields.io/badge/Agentic-AI-FF5722)
![LLM Powered](https://img.shields.io/badge/Powered%20by-LLM-9C27B0)
![Snowflake](https://img.shields.io/badge/Database-Snowflake-29B5E8)
![Docker](https://img.shields.io/badge/Containerized-Docker-2496ED)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![React](https://img.shields.io/badge/Frontend-React-61DAFB)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)

An intelligent, agentic chatbot system designed to analyze healthcare data, generate insights, and provide visualizations through natural language conversations. This project leverages advanced NLP, vector databases, and interactive data visualization to make complex healthcare information accessible. Built with FastAPI, React, Gemini AI and Pinecone vector search to bridge the gap between complex medical databases and human understanding.

## 🌟 Features

- **Natural Language Interface**: Ask complex healthcare data questions in plain English
- **Advanced Data Analysis**: SQL query generation from natural language questions
- **Interactive Visualizations**: Automatic chart generation based on query results
- **Web Search Integration**: Fallback to medical web search when database answers are insufficient
- **Real-time Communication**: WebSocket/Socket.IO support for instant responses
- **Vector Database**: Semantic search capabilities for retrieving relevant information
- **Multimodal Output**: Text responses with integrated data visualizations
- **Modern UI/UX**: Responsive, intuitive interface with professional styling

## 🏗️ Project Structure

The Healthcare Data Assistant follows a modern, modular architecture:
```
healthcare-chatbot/
├── backend/             # Python FastAPI backend
│   ├── app.py           # Main FastAPI application
│   ├── final_agentic_memory_cleaned.py  # Core assistant logic
│   ├── gen_color_plot.py               # Visualization generator
│   ├── snowflake_connector.py          # Database connection
│   └── web_search_handler.py           # Web search integration
├── frontend/            # React frontend
│   ├── public/          # Static assets
│   └── src/             # React components and logic
│       ├── components/  # UI components
│       ├── contexts/    # React contexts
│       ├── services/    # API services
│       └── styles/      # CSS styling
└── vector/              # Vector database integration
└── database/        # Vector store implementation
```

## 🔄 System Architecture

The Healthcare Data Assistant follows a sophisticated agentic architecture that enables intelligent, autonomous processing of healthcare queries:

![System Architecture](Sys%20Archi.png)

### Key Components

1. **User Interface** (React)
   - React UI for displaying query results
   - WebSocket client for real-time communication
   - Support for visualizing text responses and plots

2. **Backend Server** (FastAPI)
   - WebSocket handler for bidirectional communication
   - Application core for orchestrating the processing pipeline
   - Plots directory for storing generated visualizations

3. **Healthcare Assistant**
   - **Query Understanding**: Analyzes natural language inputs to determine intent
   - **Vector Search**: Retrieves relevant schema metadata from Pinecone
   - **SQL Generation**: Converts natural language to optimized SQL queries
   - **Database Query**: Executes queries against Snowflake database
   - **Response Generation**: Formats database results into natural language
   - **Plot Generator**: Creates visualizations based on query results

4. **Data Sources**
   - **Snowflake Database**: Primary source for healthcare data
   - **Pinecone Vector DB**: Stores schema metadata for semantic search
   - **Web Search**: Falls back to online sources when needed

### Data Flow

1. **User Query Submission** 
   - User enters natural language query through React UI
   - Query transmitted to backend via WebSocket

2. **Request Processing**
   - Backend checks rate limiting (15 API calls/minute)
   - System determines if query is conversational or data-focused
   - For data queries, main agent orchestrates processing

3. **Data Analysis Pipeline**
   - Schema information retrieved from database
   - Relevant metadata fetched via vector search in Pinecone
   - SQL query generated and executed against Snowflake
   - Results formatted into natural language response
   - Visualizations created when appropriate

4. **Response Delivery**
   - Complete response package returned to frontend via WebSocket
   - User sees results in React UI with any visualization links
   - Response stored in conversation history for context

This architecture enables seamless processing of complex healthcare queries through an intelligent, single-agent system using specialized tools.

## User Interface

### Application Frontend

<img width="1512" alt="image" src="https://github.com/user-attachments/assets/a5115317-b506-473c-b331-68b567106784" />


## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- Snowflake Database account (or modify for your preferred database)
- Optional: Docker and Docker Compose

### Environment Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/healthcare-data-assistant.git
cd healthcare-data-assistant
```

2. Create and configure environment variables:
```
# Root .env file for shared variables
cp .env.example .env

# Backend .env file
cp backend/.env.example backend/.env

# Frontend .env file
cp frontend/.env.example frontend/.env
```

3. Configure the following variables in your .env files:
```
# Snowflake credentials
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_ROLE=your_role

# API keys
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# Frontend settings
REACT_APP_API_URL=http://localhost:8000
```
## Backend Setup
1. Set up a Python virtual environment:
```
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Start the backend server:
```
python app.py
```
The FastAPI server will start at http://localhost:8000

## Frontend Setup
1. Install Node.js dependencies:
```
cd frontend
npm install
```
3. Start the frontend development server:
```
npm start
```
The React app will be available at http://localhost:3000

## Docker Deployment
Alternatively, use Docker Compose to run both backend and frontend:
```
docker-compose up
```
🧠 How It Works
The Healthcare Data Assistant is built as an agentic AI system that:

1. Receives natural language queries from users about healthcare data
2. Processes queries using LLMs to understand intent and extract key information
3. Generates optimized SQL queries to retrieve relevant healthcare data
4. Creates visualizations when appropriate based on the data returned
5. Falls back to web search when database information is insufficient
6. Maintains conversation context to enable follow-up questions

🤖 Example Queries
The system can handle a wide range of healthcare queries such as:

- "What is the average cost of a knee replacement surgery?"
- "Show me a bar chart of the top 5 most common procedures in cardiology"
- "What is the distribution of hospital stays by age group?"
- "What's the trend of COVID-19 hospitalizations over the past year?"
- "Compare the costs of different types of heart surgeries"
- "What are the most common diagnoses for patients over 65?"
- "Search the web for new cancer treatment research"

🧑‍💻 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an issue.

1. Fork the repository
2. Create a feature branch: ```git checkout -b feature/amazing-feature```
3. Commit your changes: ```git commit -m 'Add amazing feature'```
4. Push to the branch: ```git push origin feature/amazing-feature```
5. Open a Pull Request

🔜 Future Work

- Add multi-agent architecture for more specialized reasoning
- Implement continuous learning from user interactions
- Expand to support more healthcare data sources
- Add user authentication and role-based access
- Setting up containerized deployment with Kubernetes for scalability
- Support data input through file uploads
- Enhance visualization capabilities with interactive charts

🌐 Acknowledgments

- This project leverages various open-source technologies including React, FastAPI, and more.
- Special thanks to the healthcare data community for insights and test data.

📝 License
- This project is licensed under the MIT License - see the LICENSE file for details.
