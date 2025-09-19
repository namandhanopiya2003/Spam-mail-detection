import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import preprocessing
from suggestor.predict_suggestions import predict_suggestions

# Loads tokenizer for text processing
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Loads the trained spam detection model
spam_model = tf.keras.models.load_model('model/spam_detector_model.h5')
# Maximum sequence length for padding
MAX_LEN = 120

# Loads the reason classifier model and associated tools
reason_model = joblib.load('model/reason_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/reason_label_encoder.pkl')

def predict_message(message):
    # Cleans and lemmatize the input message
    cleaned = preprocessing.clean_text(message)
    lemmatized = preprocessing.lemmatize_text(cleaned)

    # Converts text into sequence and pad it for model input
    seq = tokenizer.texts_to_sequences([lemmatized])
    padded = pad_sequences(seq, padding='post', maxlen=MAX_LEN)

    # Predicts the probability of the message being spam
    spam_prob = spam_model.predict(padded, verbose=0)[0][0]

    # Determines if the message is spam or not based on threshold
    is_spam = spam_prob > 0.5
    label = 'Spam' if is_spam else 'Not Spam'
    confidence = spam_prob if is_spam else 1 - spam_prob

    # Output the prediction and confidence
    print("==========================================")
    print(f">>> Message: {message}")
    print(f">>> Prediction: {label} ({confidence * 100:.2f}% confidence)")

    if is_spam:
        # If spam, predicts the spam type and suggested actions
        spam_type, actions = predict_suggestions(message)
        print(f"\n>>> Detected Spam Type: {spam_type}")

        # Transforms the message and predict the reason for the spam
        vec = vectorizer.transform([message])
        pred = reason_model.predict(vec)[0]
        reason = label_encoder.inverse_transform([pred])[0]
        print(f">>> Predicted Reason: {reason}")

        # Prints suggested actions
        print(">>> Suggested Actions:")
        for act in actions:
            print(f"   - {act}")
    else:
        # If not spam, no further action needed
        print(">>> No further action needed.")
    print("==========================================\n")

if __name__ == "__main__":
    # List of test messages to evaluate predictions
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

    # Loops through each message and predict its status
    for msg in test_messages:
        predict_message(msg)

