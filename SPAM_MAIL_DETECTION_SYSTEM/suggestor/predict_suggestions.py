import joblib

def load_models():
    type_model = joblib.load('model/spam_type_classifier.pkl')
    action_model = joblib.load('model/action_recommender.pkl')
    label_binarizer = joblib.load('model/action_labels.pkl')
    return type_model, action_model, label_binarizer

def predict_suggestions(message_text):
    type_model, action_model, lb = load_models()

    spam_type = type_model.predict([message_text])[0]
    action_pred = action_model.predict([message_text])
    actions = lb.inverse_transform(action_pred)[0] 
    
    return spam_type, list(actions)
