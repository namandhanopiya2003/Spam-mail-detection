import joblib

model = joblib.load('model/reason_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
label_encoder = joblib.load('model/reason_label_encoder.pkl')

def predict_reason(message: str) -> str:
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    reason = label_encoder.inverse_transform([pred])[0]
    return reason

if __name__ == "__main__":
    while True:
        msg = input("\nEnter a spam message (or 'exit'): ")
        if msg.lower() == 'exit':
            break
        reason = predict_reason(msg)
        print(f"<!> Predicted Reason: {reason}")
