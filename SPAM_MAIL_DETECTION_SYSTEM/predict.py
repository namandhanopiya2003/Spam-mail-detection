import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import preprocessing
from suggestor.predict_suggestions import predict_suggestions

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

spam_model = tf.keras.models.load_model('model/spam_detector_model.h5')
MAX_LEN = 120

reason_model = joblib.load('model/reason_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/reason_label_encoder.pkl')

def predict_message(message):

    cleaned = preprocessing.clean_text(message)
    lemmatized = preprocessing.lemmatize_text(cleaned)

    seq = tokenizer.texts_to_sequences([lemmatized])
    padded = pad_sequences(seq, padding='post', maxlen=MAX_LEN)
    spam_prob = spam_model.predict(padded, verbose=0)[0][0]

    is_spam = spam_prob > 0.5
    label = 'Spam' if is_spam else 'Not Spam'
    confidence = spam_prob if is_spam else 1 - spam_prob

    print("==========================================")
    print(f">>> Message: {message}")
    print(f">>> Prediction: {label} ({confidence * 100:.2f}% confidence)")

    if is_spam:
        spam_type, actions = predict_suggestions(message)
        print(f"\n>>> Detected Spam Type: {spam_type}")

        vec = vectorizer.transform([message])
        pred = reason_model.predict(vec)[0]
        reason = label_encoder.inverse_transform([pred])[0]
        print(f">>> Predicted Reason: {reason}")

        print(">>> Suggested Actions:")
        for act in actions:
            print(f"   - {act}")
    else:
        print(">>> No further action needed.")
    print("==========================================\n")

if __name__ == "__main__":
    test_messages = [
        "Win a brand new iPhone for just $1. Claim now",
        "Congratulations! You’ve won a $1000 Walmart gift card. Go to http://spammy.link now!",
        "I will meet you in office.",
        "How are you?",
        "Congratulations! You've been selected for a special offer.",
        "Please verify your account details immediately to avoid suspension.",
        "Update your software to the latest version for better security",
        "Your package is out for delivery, no action needed.",
        "Meeting rescheduled to 3 PM, please confirm your availability.",
        "Click here to speed up your system instantly."
    ]
    for msg in test_messages:
        predict_message(msg)
