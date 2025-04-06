import os
import subprocess
import sys

def create_project_structure():
    """
    Create project directory structure
    """
    directories = [
        'src/agents',
        'src/database',
        'src/utils',
        'data/metadata',
        'data/processed',
        'notebooks',
        'tests',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_requirements_file():
    """
    Create requirements.txt with necessary dependencies
    """
    requirements = [
        "pandas",
        "numpy",
        "sentence-transformers",
        "scikit-learn",
        "pinecone-client",
        "snowflake-connector-python",
        "flask",
        "openai",
        "tiktoken",
        "python-dotenv",
        "loguru",
        "pytest",
        "matplotlib",
        "seaborn"
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("Created requirements.txt")

def create_environment_file():
    """
    Create .env template file
    """
    env_contents = """
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter

# Snowflake Credentials
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/application.log
"""
    
    with open('.env', 'w') as f:
        f.write(env_contents)
    
    print("Created .env template file")

def create_gitignore():
    """
    Create .gitignore file
    """
    gitignore_contents = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# Logs
logs/
*.log

# Data
data/processed/
*.csv
*.xlsx

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_contents)
    
    print("Created .gitignore")

def create_readme():
    """
    Create README.md with project information
    """
    readme_contents = """
# Healthcare Data Insights Project

## Project Overview
A hybrid AI-powered system for extracting insights from healthcare data using vector embeddings and LLM agents.

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip
- virtualenv (recommended)

### Installation Steps
1. Clone the repository
2. Create a virtual environment
"""

def main():
    create_project_structure()
    create_requirements_file()
    create_environment_file()
    create_gitignore()
    create_readme()

if __name__ == "__main__":
    main()