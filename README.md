# Weather & Document RAG Agent

An intelligent agent built with LangChain, LangGraph, and LangSmith that can fetch real-time weather data and answer questions from documents using Retrieval-Augmented Generation (RAG).

## Overview

This project demonstrates a complete AI pipeline that:

1. Uses **LangGraph** to build an agentic workflow that decides whether to fetch weather data or query documents
2. Retrieves **real-time weather data** from OpenWeatherMap API
3. Implements **RAG (Retrieval-Augmented Generation)** to answer questions from PDF documents
4. Stores document embeddings in a **Qdrant vector database**
5. Processes data using **OpenAI's language models** via LangChain
6. Evaluates responses with **LangSmith**
7. Presents a user-friendly interface with **Streamlit**

## Features

- **Intelligent Query Classification**: Automatically determines whether a query requires weather data or document information
- **Real-time Weather Data**: Fetches current weather conditions for any city worldwide
- **Document Processing Pipeline**:
  - PDF loading and chunking
  - Text embedding generation
  - Vector storage in Qdrant
  - Semantic retrieval for relevant information
- **Comprehensive Evaluation**: Uses LangSmith to evaluate LLM responses for accuracy and relevance
- **Clean, Modular Architecture**: Well-structured code with proper separation of concerns
- **Full Test Coverage**: Unit tests for API handling, LLM processing, and retrieval logic
- **Interactive UI**: Simple chat interface for interacting with the agent

## Installation & Setup

### Prerequisites

- Python 3.8+
- Docker (for running Qdrant locally)
- API keys:
  - OpenWeatherMap API key
  - OpenAI API key
  - LangSmith API key

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/shregar1/weatheragent.git
cd weatheragent
```

2. **Set up virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the root directory with the following:

```
OPENWEATHER_API_KEY=your_openweather_api_key
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
QDRANT_URL=http://localhost:6333
```

5. **Start Qdrant**

```bash
docker pull qdrant/qdrant
```

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

6. **Run the application**

```bash
streamlit run app.py
```

## Usage Guide

### Working with Documents

1. **Upload PDF documents**: Use the file uploader in the sidebar to upload PDF files
2. **Document Processing**: The system automatically:
   - Loads PDF content
   - Splits documents into chunks
   - Generates embeddings
   - Stores them in the vector database

### Asking Questions

The agent can handle two types of queries:

1. **Weather queries**: Ask about weather in specific cities
   - Example: "What's the weather like in Tokyo today?"
   - Example: "Is it raining in New York?"

2. **Document queries**: Ask questions about uploaded documents
   - Example: "What are the main points in the climate change report?"
   - Example: "Summarize the key findings in the document"

### System Response

The system will:
1. Classify your query type (weather or document)
2. Fetch appropriate data (weather API or document chunks)
3. Generate a comprehensive response using the LLM
4. Track and evaluate the response with LangSmith

## Code Structure

```
weather-rag-agent/
├── .env                    # Environment variables
├── app.py                  # Streamlit UI application
├── config.py               # Configuration settings
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── data/                   # Store PDFs
├── tests/                  # Test cases
│   ├── __init__.py
│   ├── test_api.py         # Weather API tests
│   ├── test_llm.py         # LLM processing tests
│   └── test_retrieval.py   # Vector database tests
└── src/
    ├── __init__.py
    ├── agent/              # LangGraph implementation
    │   ├── __init__.py
    │   ├── graph.py        # Agent graph definition
    │   └── nodes.py        # Node implementations
    ├── api/                # External API integrations
    │   ├── __init__.py
    │   └── weather.py      # Weather API interface
    ├── document/           # Document processing
    │   ├── __init__.py
    │   ├── loader.py       # PDF loading
    │   └── processor.py    # Document chunking
    ├── embedding/          # Vector database operations
    │   ├── __init__.py
    │   └── vectordb.py     # Qdrant integration
    └── llm/                # LLM integrations
        ├── __init__.py
        ├── chain.py        # LangChain setup
        └── evaluation.py   # LangSmith evaluation
```

## Implementation Details

### LangGraph Agent Flow

The agent uses LangGraph to implement a decision-making workflow:

1. **classify_query node**: Determines if the query is about weather or documents
2. **get_weather node**: Fetches and processes weather data if needed
3. **query_document node**: Retrieves relevant document chunks if needed
4. **generate_response node**: Formulates the final response

### Document Processing

1. **Loading**: Uses PyPDFLoader to extract text from PDFs
2. **Chunking**: Implements RecursiveCharacterTextSplitter for semantic chunking
3. **Embedding**: Uses OpenAI's embedding model to create vector representations
4. **Storage**: Stores embeddings in Qdrant vector database

### RAG Implementation

The RAG pipeline:
1. Takes a user question
2. Retrieves the most relevant document chunks using semantic similarity
3. Combines these chunks with the question in a prompt
4. Generates a comprehensive answer using the LLM

### Weather Data Processing

1. Fetches current weather data from OpenWeatherMap API
2. Formats data into a readable structure
3. Uses LLM to answer specific questions about the weather conditions

### Evaluation with LangSmith

The system tracks and evaluates LLM responses using LangSmith:
1. Records input, output, and intermediate steps
2. Evaluates responses based on correctness and relevance
3. Provides metrics for model performance

## Testing

Run the test suite with:

```bash
python -m unittest discover tests
```

The tests cover:
- Weather API interaction
- LLM chain functionality
- Vector database operations

## LangSmith Integration

This project uses LangSmith for:
1. **Tracing**: Recording all steps in the agent workflow
2. **Evaluation**: Assessing response quality
3. **Debugging**: Identifying issues in the pipeline

To view traces and evaluations:
1. Log into your LangSmith account
2. Navigate to the "rag-weather-agent" project
3. Explore traces, inputs, outputs, and evaluation metrics

## Demo Video

A Loom video explaining the implementation and demonstrating the application is available [here](https://drive.google.com/file/d/1SSQn_1Gn9vNOJUMKmNVoZ9wFdBIjpK9m/view?usp=drive_link).

## Future Improvements

Potential enhancements for this project:
- Add support for more document formats (DOCX, TXT, etc.)
- Implement memory to handle follow-up questions
- Add multi-document correlation for more comprehensive answers
- Integrate with more weather data sources for historical data
- Implement advanced LangSmith feedback collection

## License

MIT

## Contact

For questions or feedback, please contact:
- Email: sengarsinghshreyansh@gmail.com
- GitHub: [shregar1](https://github.com/shregar1)