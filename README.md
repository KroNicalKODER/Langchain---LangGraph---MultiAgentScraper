# Research Assistant with LangGraph

A powerful research assistant that uses LangGraph for workflow management, Tavily for web scraping, and Ollama for LLM processing. The system provides detailed research responses with a transparent **thinking process**.

## Features

- 🤖 Interactive chat interface
- 🔍 Real-time web research
- 🧠 Transparent thinking process
- 📚 Parallel information gathering
- 💡 Context-aware responses

## Prerequisites

- Python 3.10 or higher
- Conda (recommended) or virtualenv
- Ollama installed and running with the Llama3.2 model
- Tavily API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KroNicalKODER/Langchain---LangGraph---MultiAgentScraper.git
cd Langchain---LangGraph---MultiAgentScraper
```

2. Create and activate a conda environment:
```bash
conda create -p venv python=3.10
conda activate ./venv
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
TAVILY_API_KEY=your_tavily_api_key
```

## Running Ollama

Before starting the application, ensure Ollama is running with the DeepSeek model:

```bash
# Start Ollama service
ollama serve

# In a separate terminal, pull the llama3.2 model
ollama pull llama3.2
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## Usage

1. Enter your research query in the chat input
2. The assistant will:
   - Gather initial context
   - Break down the query into sub-queries
   - Search for information in parallel
   - Combine and analyze the results
   - Provide a comprehensive response

3. Toggle the "Show Thinking Process" checkbox to view the detailed research steps

## Images
![image](https://github.com/user-attachments/assets/b6c7ecc6-dbd9-4bb7-af78-760177c15a21)
![image](https://github.com/user-attachments/assets/f3137dd4-50d8-456c-ba33-cd8948b71efc)
![image](https://github.com/user-attachments/assets/71a450a9-c01b-406d-824e-04bdcca14e9f)



