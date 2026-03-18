# How Rag Works

Rag is a technique that aims to fix the limitation of LLMs by supplementing it with information retrieval for domain specific information.

The pipeline goes like:
1. data collection
2. chunking
3. embeddings
4. query


## Data sources
Commonly in a form of pdf documents for business applications. but can use text
data sources include: 
- user manuals
- datasheets
- faqs


For RAG to work, you would need to have:
embeddings model
