"""
Author: Amisha

"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# Ensure the preprocessing function is available
from app.preprocess1 import preprocess_text

# Load the data
# data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
data_path = 'data/data.csv'
df = pd.read_csv(data_path, sep='|')

# Ensure no need to preprocess again if already done during saving
# df['question'] = df['question'].apply(preprocess_text)
# df['answer'] = df['answer'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answer'], test_size=0.2, random_state=42)

# Train the answer rating model
vectorizer = TfidfVectorizer()
X_answers = vectorizer.fit_transform(df['answer'])
rating_model = RandomForestRegressor()
rating_model.fit(X_answers, df['score'])

# Train the question-answer matching model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])
pipeline.fit(X_train, y_train)

# Save the models
model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
joblib.dump(rating_model, os.path.join(model_dir, 'rating_model.pkl'))
joblib.dump(pipeline, os.path.join(model_dir, 'pipeline.pkl'))
