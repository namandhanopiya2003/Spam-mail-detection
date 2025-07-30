# Import required libraries
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Cleans input text by: Lowercasing, Removing URLs, Removing special characters and numbers, Removing extra whitespaces
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    # Converts each word in the input text to its root form (Example: 'running' -> 'run', 'better' -> 'good', etc.)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def load_and_preprocess_data(filepath):
    # Loads dataset from a CSV file and applies text preprocessing
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['message'] = df['message'].apply(clean_text)
    df['message'] = df['message'].apply(lemmatize_text)
    return df
