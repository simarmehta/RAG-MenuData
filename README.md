## RAG Chatbot Application

This repository implements a complete Retrieval-Augmented Generation (RAG) pipeline for a conversational chatbot. The system integrates structured restaurant menu data with external cuisine context to answer user queries using vector search and GPT-4 prompt-based generation. The project is composed of three main components:

### Data Ingestion & Indexing:
- Processes internal CSV data and external web content
- Converts text into embeddings using SentenceTransformer
- Builds vector indexes (with FAISS)

### Chatbot Backend:
- A FastAPI application that receives a query along with conversation history
- Retrieves internal and external records, applies deduplication and classification
- Assembles a prompt (including conversation history), calls GPT-4, and returns an answer with updated conversation history and reference information

### Chatbot UI:
- A Gradio-based frontend that provides an interactive chat interface
- Maintains conversation context
- Displays the conversation history and shows internal/external references in a formatted section

## Table of Contents
- Repository Structure
- Installation
- Usage
- Data Ingestion & Indexing
- Chatbot Backend
- Chatbot UI
- Pipeline Architecture


## Repository Structure
```
.
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── run.py    # Notebook/script for data ingestion and indexing
├── chatbot_app.py                   # FastAPI backend for the RAG pipeline
├── chatbot_ui.py                    # Gradio frontend for the conversational chatbot
├── data_sentences.jsonl             # Processed internal sentences (JSONL file) # note once you run run.ipynb you will have these data files
├── output.json                      # Grouped internal data by restaurant
├── metadata_mapping.pkl             # Pickled metadata mapping for internal data
├── faiss_index.bin                  # FAISS index file (if using FAISS; see notes below for Milvus)
├── multiples_scraped_data.json      # Scraped external data in JSON format
├── faiss_external_mapping.json      # External metadata mapping file
└── (Additional scripts or notebooks as needed)
```
Note once you run run.ipynb you will have the index and mapping files

## Installation
### Clone the Repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### Install Python Dependencies and Install libraries in requirements.txt:
Install the required packages:
```bash
pip install faiss-cpu torch transformers sentence-transformers pandas numpy beautifulsoup4 scrapy fastapi uvicorn requests
pip install -r requirements.txt

```


### Set Up API Keys:
Replace the placeholder OpenAI API key in `chatbot_app.py` with your actual API key or set it as an environment variable.

## Usage
### Data Ingestion & Indexing
**Internal Data:**
- The CSV file (e.g., `Sample Ingredients File - MenuData Mission.csv`) is loaded .
- Column names are normalized, and each row is converted to a descriptive sentence.
- The sentences are saved into a JSONL file and then grouped by restaurant name.
- The SentenceTransformer model embeds each sentence, and the resulting vectors are indexed:
 
**External Data:**
- External websites (e.g., culinary blogs or Wikipedia) are scraped using BeautifulSoup to extract titles and paragraphs.
- The extracted paragraphs are embedded with SentenceTransformer.

Run the data ingestion notebook or script (e.g., `run.py` or `data_ingestion.ipynb`) to build these indexes.

### Chatbot Backend
Start the FastAPI server by running:
```bash
python chatbot_app.py
```
This launches the backend server on `http://0.0.0.0:8000`.

### Chatbot UI
To launch the Gradio-based frontend, run:
```bash
python chatbot_ui.py
```
This interactive interface lets you type queries, shows the conversation history, and displays internal/external references.

## Pipeline Architecture
https://docs.google.com/document/d/1obYyidaZgd8Iu291ZbEqRCqkicxQqoLFhYCli0x9qqM/edit?usp=sharing


