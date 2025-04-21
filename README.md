# Research Assistant with LangGraph

A powerful research assistant that uses LangGraph for workflow management, Tavily for web scraping, and Ollama for LLM processing. The system provides detailed research responses with a transparent thinking process.

## Features

- 🤖 Interactive chat interface
- 🔍 Real-time web research
- 🧠 Transparent thinking process
- 📚 Parallel information gathering
- 💡 Context-aware responses

## Prerequisites

- Python 3.10 or higher
- Conda (recommended) or virtualenv
- Ollama installed and running with the DeepSeek model
- Tavily API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
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

## Requirements

The project requires the following packages (automatically installed via requirements.txt):
```
langchain>=0.1.0
langgraph>=0.0.10
streamlit>=1.30.0
tavily-python>=0.2.0
python-dotenv>=1.0.0
```

## Running Ollama

Before starting the application, ensure Ollama is running with the DeepSeek model:

```bash
# Start Ollama service
ollama serve

# In a separate terminal, pull the DeepSeek model
ollama pull deepseek-coder:33b
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

## Project Structure

```
.
├── app.py                 # Streamlit application
├── research_graph.py      # LangGraph workflow implementation
├── requirements.txt       # Project dependencies
├── .env                  # Environment variables
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow management
- [Tavily](https://tavily.com/) for web search capabilities
- [Ollama](https://ollama.ai/) for local LLM processing 