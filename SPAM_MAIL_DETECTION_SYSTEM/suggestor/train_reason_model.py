# import required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# loads labeled dataset for spam reasons
df = pd.read_csv('data/spam_reason_labeled.csv')
df.dropna(inplace=True)

# separates features and target labels
X = df['message']
y = df['reason']

# encode string labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# initialized tf-idf vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# created pipeline with vectorizer and logistic regression model
clf = Pipeline([
    ('tfidf', vectorizer),
    ('model', LogisticRegression(max_iter=1000))
])

# trained the model
clf.fit(X, y_encoded)

os.makedirs('model', exist_ok=True)
# saves trained model, vectorizer, and label encoder
joblib.dump(clf.named_steps['model'], 'model/reason_classifier.pkl')
joblib.dump(clf.named_steps['tfidf'], 'model/vectorizer.pkl')
joblib.dump(label_encoder, 'model/reason_label_encoder.pkl')

# evaluates model performance on training data
acc = clf.score(X, y_encoded)
print(f">>> Training completed. Accuracy on training data: {acc:.4f}")

