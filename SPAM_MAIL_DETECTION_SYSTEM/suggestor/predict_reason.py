# import required libraries
import joblib

# loads trained model, vectorizer, and label encoder
model = joblib.load('model/reason_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/reason_label_encoder.pkl')

# predicts the reason behind a given spam message
def predict_reason(message: str) -> str:

    # vectorized the input message
    vec = vectorizer.transform([message])
    # predicted the encoded label
    pred = model.predict(vec)[0]
    # decoded the predicted label to original reason
    reason = label_encoder.inverse_transform([pred])[0]
    return reason

# runs prediction loop until user types 'exit'
if __name__ == "__main__":
    while True:
        msg = input("\nEnter a spam message (or 'exit'): ")
        if msg.lower() == 'exit':
            break
        reason = predict_reason(msg)
        print(f"<!> Predicted Reason: {reason}")

