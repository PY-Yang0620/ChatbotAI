# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 02:49:09 2024

@author: Pyang
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import warnings
import urllib3

# Disable warnings for self-signed certificates
warnings.filterwarnings("ignore", message="Connecting to 'https://localhost' using TLS with verify_certs=False is insecure")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Path to the PDF document
pdf_path = r"C:pdfPath"

# Extract text from PDF pages
def extract_text_from_pdf(pdf_path):
    text_data = []
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text = page.get_text("text")
            text_data.append(text)
    return text_data

# OCR for image extraction (if tables/images exist)
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Extract and deduplicate text sections for knowledge base
extracted_text = extract_text_from_pdf(pdf_path)
unique_text_data = list({text for text in extracted_text})  # Remove duplicates
knowledge_base = [
    {"text": section, "metadata": {"section": "Risk Classification", "category": "High-Risk"}}
    for section in unique_text_data
]

# Connect to Elasticsearch over HTTPS
es = Elasticsearch(
    "https://localhost",
    basic_auth=("elastic", "password"),  # Replace with your actual password
    verify_certs=False  # Disable SSL verification for self-signed certificates
)

# Indexing documents
def index_data(documents):
    actions = [
        {
            "_index": "eu_ai_act",
            "_source": doc
        }
        for doc in documents
    ]
    bulk(es, actions)

# Indexing unique knowledge base
index_data(knowledge_base)

# Querying Elasticsearch with deduplication of results
def retrieve_from_elasticsearch(index_name, query, size=5):
    response = es.search(index=index_name, query={"match": {"text": query}}, size=size)
    unique_results = []
    seen_texts = set()

    for hit in response["hits"]["hits"]:
        text = hit["_source"]["text"]
        if text not in seen_texts:
            unique_results.append(hit["_source"])
            seen_texts.add(text)

    return unique_results

# Example query and displaying results
query = "What are high-risk AI systems?"
retrieved_docs = retrieve_from_elasticsearch("eu_ai_act", query)

# Display the results - limiting each document's text for easier readability
for i, doc in enumerate(retrieved_docs[:3]):  # Display only first 3 for brevity
    print(f"Document {i+1}")
    print("Text:", doc["text"][:200] + "...")  # Print only first 200 characters
    print("Category:", doc["metadata"]["category"])
    print("=" * 50)

# Optional: Print the entire retrieved_docs list if needed for debugging
print(retrieved_docs[:3])  # Only display the first 3 entries for verification






def retrieve_documents(query, es, index_name="eu_ai_act", size=5):
    response = es.search(index=index_name, query={"match": {"text": query}}, size=size)
    unique_results = []
    seen_texts = set()

    for hit in response["hits"]["hits"]:
        text = hit["_source"]["text"]
        if text not in seen_texts:
            unique_results.append(hit["_source"]["text"])
            seen_texts.add(text)

    return unique_results





import openai

# Set up the API key
openai.api_key = 'key'

def generate_response_with_gpt(retrieved_docs, user_query):
    # Combine retrieved documents into a single context
    context = "\n\n".join(retrieved_docs)
    
    # Define the prompt to include the context and user query
    prompt = f"Context:\n{context}\n\nAnswer the following question based on the context: {user_query}"

    # Call OpenAI API to generate response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are an assistant well-versed in the EU AI Act."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']





def rag_pipeline(query, es, index_name="eu_ai_act"):
    # Step 1: Retrieve relevant documents from Elasticsearch
    retrieved_docs = retrieve_documents(query, es, index_name=index_name)
    
    # Step 2: Generate response with OpenAI based on retrieved documents
    if retrieved_docs:
        generated_response = generate_response_with_gpt(retrieved_docs, query)
        return generated_response
    else:
        return "No relevant information found for your query."

# Example usage
user_query = "What are high-risk AI systems under the EU AI Act?"
response = rag_pipeline(user_query, es)
print(response)
