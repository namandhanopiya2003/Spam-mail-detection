# Import required libraries
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Loads labeled dataset for training suggestion models
df = pd.read_csv('data/spam_suggestions_labeled.csv')

# Drops rows with missing values in critical columns
df = df.dropna(subset=['message_text', 'spam_type', 'suggested_actions'])

# Converts semicolon-separated actions into a list format
df['suggested_actions'] = df['suggested_actions'].apply(lambda x: [action.strip() for action in x.split(';')])

# Defined features and target for spam type classification
X_text = df['message_text']
y_type = df['spam_type']

# Creates a pipeline for spam type classification using TF-IDF + Random Forest
type_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Trained and save the spam type classification model
type_pipeline.fit(X_text, y_type)
joblib.dump(type_pipeline, 'model/spam_type_classifier.pkl')
print(">>> Spam type classifier saved.")

# Prepared multi-label binarizer for suggested actions
mlb = MultiLabelBinarizer()
y_actions = mlb.fit_transform(df['suggested_actions'])

# Created a pipeline for action recommendation
action_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# Trains and save the action recommendation model
action_pipeline.fit(X_text, y_actions)
joblib.dump(action_pipeline, 'model/action_recommender.pkl')
joblib.dump(mlb, 'model/action_labels.pkl')
print(">>> Action recommender and label binarizer saved.")
