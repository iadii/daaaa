"""
Author: Amisha & Aditya
Date: 2024-05-30
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Ensure the preprocessing function is available
from app.preprocess import preprocess_text

# Construct the data file path
data_path = 'data/data.csv'

# Check if the data file exists
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"Data file not found at path: {data_path}")

# Load the data
df = pd.read_csv(data_path, sep='|')

# Verify the columns in the dataframe
if 'answer' not in df.columns:
    raise KeyError("'answer' column not found in the dataframe.")
if df['answer'].isnull().any():
    raise ValueError("Missing values found in 'answer' column.")

# Optionally preprocess the data
# If the preprocessing needs to be applied, uncomment the following lines
df['question'] = df['question'].apply(preprocess_text)
df['answer'] = df['answer'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answer'], test_size=0.2, random_state=42)

# Train the answer rating model
vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')  # Adjust token_pattern to include single characters
X_answers = vectorizer.fit_transform(df['answer'])
rating_model = RandomForestRegressor()
rating_model.fit(X_answers, df['score'])

# Train the question-answer matching model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')),  # Adjust token_pattern to include single characters
    ('clf', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Save the models
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump(rating_model, os.path.join(model_dir, 'rating_model.pkl'))
joblib.dump(pipeline, os.path.join(model_dir, 'pipeline.pkl'))

model = RandomForestClassifier()
model.fit(X_train, y_train)
print("Model accuracy: ", model.score(X_test, y_test))


print("Models saved successfully!")
