"""
Author: Amisha & Aditya
Date: 2024-05-30
"""
# Import necessary libraries
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import json
import re
import os

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define preprocessing function
def preprocess_text(text):
    text = str(text)
    words = word_tokenize(text.lower())
    pattern = re.compile("^[a-zA-Z0-9+\-^*/%=]+$")
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and pattern.match(word)]
    return ' '.join(words)

# Define path to JSON file
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'config.json')

# Load JSON data
with open(data_path, 'r') as file:
    data = json.load(file)

# Ensure JSON data is a list
if not isinstance(data, list):
    raise ValueError("JSON data should be a list of dictionaries")

# Initialize numpy arrays
num_samples = len(data)
questions = np.empty(num_samples, dtype=object)
answers = np.empty(num_samples, dtype=object)
scores = np.empty(num_samples, dtype=object)

# Iterate over each item in the data list
for idx, item in enumerate(data):
    if "question" in item and "answer" in item and "mark" in item:
        question = item["question"]["ques"][0] if "ques" in item["question"] else ""
        answer = item["answer"][0] if item["answer"] else ""
        score = item["mark"]
        questions[idx] = question
        answers[idx] = answer
        scores[idx] = score
    else:
        raise ValueError("Each item in JSON data should have 'question', 'answer', and 'mark' keys")

# Create DataFrame
df = pd.DataFrame({'question': questions, 'answer': answers, 'score': scores})

# Apply preprocessing
df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)
df['score'] = df['score'].astype(float)  # Ensure the score column is of type float

# Save DataFrame to CSV
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
df.to_csv(output_path, sep='|', index=False)

print(df)
