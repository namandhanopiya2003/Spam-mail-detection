# Import required libraries
import joblib

def load_models():
    # Loads trained models and label binarizer
    type_model = joblib.load('model/spam_type_classifier.pkl')
    action_model = joblib.load('model/action_recommender.pkl')
    label_binarizer = joblib.load('model/action_labels.pkl')
    return type_model, action_model, label_binarizer

# Predicts the spam type and recommended actions for the input message
def predict_suggestions(message_text):

    # Loads models and label binarizer
    type_model, action_model, lb = load_models()

    # Predicts type of spam (e.g., phishing, marketing)
    spam_type = type_model.predict([message_text])[0]

    # Predicts recommended actions (e.g., 'block sender', 'report')
    action_pred = action_model.predict([message_text])

    # Converts predictions back to readable labels
    actions = lb.inverse_transform(action_pred)[0] 

    # Returns spam type and list of suggested actions
    return spam_type, list(actions)
    
