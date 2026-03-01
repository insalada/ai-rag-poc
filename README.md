# ai-rag-poc

An IA POC that simulates a company chatbot implementing a pretty basic RAG based on langchain embeddings, vectorDB and OpenAI.


# How to run

Use the environment:

```python -m venv .venv && pip install -r requirements.txt```

Run:

```python3 rag-poc.py```


# How it works

This script just loads a couple of hardcoded lines from a made-up company documents, splits in chunks and stores in Chroma

When a question is made, it is retrieved by similarity and added as context for the LLM.