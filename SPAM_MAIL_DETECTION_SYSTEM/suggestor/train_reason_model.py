import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

df = pd.read_csv('data/spam_reason_labeled.csv')
df.dropna(inplace=True)

X = df['message']
y = df['reason']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

vectorizer = TfidfVectorizer(max_features=5000)

clf = Pipeline([
    ('tfidf', vectorizer),
    ('model', LogisticRegression(max_iter=1000))
])

clf.fit(X, y_encoded)

os.makedirs('model', exist_ok=True)
joblib.dump(clf.named_steps['model'], 'model/reason_classifier.pkl')
joblib.dump(clf.named_steps['tfidf'], 'model/vectorizer.pkl')
joblib.dump(label_encoder, 'model/reason_label_encoder.pkl')

acc = clf.score(X, y_encoded)
print(f">>> Training completed. Accuracy on training data: {acc:.4f}")
