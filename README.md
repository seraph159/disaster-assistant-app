# Disaster Assistant App

A Streamlit application that provides information and guidance during disaster situations using LLM and RAG components.

## Features

- **Emergency Chat**: Get answers to your questions about disaster preparedness and response
- **Resources**: Access important emergency contacts and resources
- **RAG-powered responses**: Uses document retrieval to provide more accurate information

## Setup Instructions

### Prerequisites

- Python 3.11
- Pipenv
- Groq API key

### Installation

1. Clone this repository
2. Navigate to the project directory
3. Install dependencies using pipenv:

```bash
pipenv install
```

### Setting up your API key

Set your Groq API key as an environment variable:

```bash
# On Windows
set GROQ_API_KEY=your_api_key_here

# On macOS/Linux
export GROQ_API_KEY=your_api_key_here
```

### Running the application

```bash
pipenv run streamlit run disaster_assistant.py
```

## Adding Disaster Documents

Place PDF documents with disaster-related information in the `disaster_docs` directory to enhance the RAG capabilities of the application.

## Important Note

In a real emergency, always contact local emergency services first!