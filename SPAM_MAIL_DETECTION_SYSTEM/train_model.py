import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import preprocessing


def main():
    if not os.path.exists('data/sms_spam.csv'):
        print("<!><!> Dataset not found: data/sms_spam.csv")
        return

    print(">>>> Loading and preprocessing data...")
    df = preprocessing.load_and_preprocess_data('data/sms_spam.csv')
    texts = df['message'].values
    labels = df['label'].values

    print(">>>> Encoding labels...")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    print(">>>> Tokenizing text...")
    vocab_size = 10000
    max_len = 120
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding='post', maxlen=max_len)

    print(">>>> Building model...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(">>>> Training model...")
    model.fit(padded, labels_encoded, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

    print(">>>> Saving model and tokenizer...")
    os.makedirs('model', exist_ok=True)
    model.save('model/spam_detector_model.h5')

    with open('model/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    print(">>>> Evaluating model...")
    predictions = (model.predict(padded) > 0.5).astype("int32")
    print(classification_report(labels_encoded, predictions))

    print(">>>> Training completed and model saved successfully.")


if __name__ == "__main__":
    main()
