# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 00:43:23 2024

@author: Pyang
"""

import pdfplumber

# Load PDF
with pdfplumber.open(r"C:\Users\Pyang\Downloads\BTTT\Project\EPRS_BRI(2021)698792_EN (1).pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

print(text[:500])  # Print first 500 characters

##TEXT EXPLORATION##
from collections import Counter
import re

# Tokenize the text into words (ignoring case and special characters)
words = re.findall(r'\b\w+\b', text.lower())

# Check the length of the text
print(f"Total characters in document: {len(text)}")
print(f"Total words in document: {len(words)}")

# Count the most common words
word_counts = Counter(words)
print(word_counts.most_common(10))  # Top 10 most frequent words

import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Remove stopwords from the word list
filtered_words = [word for word in words if word not in stop_words]

# Re-count the most common words after filtering
filtered_word_counts = Counter(filtered_words)
print(filtered_word_counts.most_common(10))

##VISUALIZE WROD FREQUENCIES##
import matplotlib.pyplot as plt

# Get the 10 most common words
common_words = filtered_word_counts.most_common(10)

# Separate words and their frequencies
words, counts = zip(*common_words)

# Plot
plt.figure(figsize=(10,6))
plt.bar(words, counts)
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

##EXTRACT AND EXPLORE TABLES,CHARTS,OR GRAPHS##
with pdfplumber.open(r"C:\Users\Pyang\Downloads\BTTT\Project\EPRS_BRI(2021)698792_EN (1).pdf") as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        for table in tables:
            print(f"Table found on page {page_num}:")
            for row in table:
                print(row)

##OVERALL SENTIMENT OF DOC##
from textblob import TextBlob

# Perform sentiment analysis
blob = TextBlob(text)

# Output sentiment polarity (-1 = negative, 1 = positive)
print(f"Sentiment Polarity: {blob.sentiment.polarity}")
print(f"Sentiment Subjectivity: {blob.sentiment.subjectivity}")

##KEY PEOPLE,ORGANIZATIONS,LOCATION MENTIONED##
import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy document object
doc = nlp(text)

# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Named Entities in the document:")
for entity in entities:
    print(entity)
