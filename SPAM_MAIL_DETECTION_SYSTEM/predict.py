import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import preprocessing

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
model = tf.keras.models.load_model('model/spam_detector_model.h5')

def predict_message(message):
    cleaned = preprocessing.clean_text(message)
    lemmatized = preprocessing.lemmatize_text(cleaned)
    seq = tokenizer.texts_to_sequences([lemmatized])
    padded = pad_sequences(seq, padding='post', maxlen=120)

    pred_prob = model.predict(padded)[0][0]
    prediction = 1 if pred_prob > 0.5 else 0
    label = 'Spam' if prediction == 1 else 'Not Spam'
    confidence = pred_prob if prediction == 1 else 1 - pred_prob

    print(f">>>> Message: {message}")
    print(f">>>> Prediction: {label} ({confidence * 100:.2f}% confidence)")

predict_message("Win a brand new iPhone for just $1. Claim now")
predict_message("Congratulations! You’ve won a $1000 Walmart gift card. Go to http://spammy.link now!")
predict_message("I will meet you in office.")
predict_message("How are you?")

predict_message("Congratulations! You've been selected for a special offer.")
predict_message("Please verify your account details immediately to avoid suspension.")
predict_message("Update your software to the latest version for better security")

predict_message("Your package is out for delivery, no action needed.")
predict_message("Meeting rescheduled to 3 PM, please confirm your availability.")