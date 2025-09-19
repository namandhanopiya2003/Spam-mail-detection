## ABOUT THIS PROJECT ==>
- This is a machine learning project that detects whether a message is *Spam* or *Not Spam* using natural language processing (NLP) and a deep learning model (LSTM). It is trained on a dataset of SMS messages and can be used to predict new messages with high accuracy.

- Spam detection is an essential part of modern communication systems. This project demonstrates a simple yet powerful approach to classify messages using a deep learning model. It leverages the LSTM layer to capture the sequence nature of text and predict the likelihood of spam.

---

## ⚙ TECHNOLOGIES USED ==>

- Python
- TensorFlow / Keras
- Pandas & NumPy
- NLTK (for stopwords & text cleaning)
- Scikit-learn (optional for evaluation)
- Jupyter / Terminal (for training & testing)

---

## 📁 PROJECT FOLDER STRUCTURE ==>

FRAUD_DETECTION_SYSTEM/<br>
├── data/<br>
│   ├── sms_spam.csv                          # Dataset with labeled SMS messages<br>
│   ├── spam_reason_labeled.csv<br>
│   └── spam_suggestions_labeled.csv          # Dataset to train suggestion model<br>
│<br>
├── model/<br>
│   ├── action_labels.pkl<br>
│   ├── action_recommender.pkl<br>
│   ├── label_encoder.pkl<br>
│   ├── reason_classifier.pkl<br>
│   ├── reason_label_encoder.pkl<br>
│   ├── spam_detector_model.h5<br>
│   ├── spam_type_classifier.pkl<br>
│   ├── tokenizer.pkl<br>
│   └── vectorizer.pkl<br>
│<br>
├── suggestor/<br>
│   ├── predict_reason.py                     # Predicts the reason (cause) behind a spam message<br>
│   ├── predict_suggestions.py                # Loads trained models and suggests actions<br>
│   ├── train_reason_model.py                 # Trains ML model that predicts the reason user might be receiving spam<br>
│   └── train_suggestion_models.py            # Script to train suggestion model<br>
│<br>
├── preprocessing.py                          # Text cleaning and preprocessing functions<br>
├── train_model.py                            # Code to train the LSTM model<br>
├── predict.py                                # Script to make predictions using trained model<br>
├── requirements.txt<br>
└── README.md                                 # Project documentation

---

## 📝 WHAT EACH FILE DOES ==>

- **data/sms_spam.csv**: The raw dataset used for training and testing. Contains labeled SMS messages.

- **preprocessing.py**: Cleans and preprocesses the text (lowercasing, removing URLs, punctuation, stopwords, etc.).

- **train_model.py**:
  - Loads and cleans the dataset.
  - Tokenizes and pads the text data.
  - Trains an LSTM-based model.
  - Saves the trained model and tokenizer.

- **predict.py**:
  - Loads the trained model and tokenizer.
  - Preprocesses a new message.
  - Predicts whether it's spam or not with confidence.

- **model/**:
  - spam_detector_model.h5: The final trained model file.
  - tokenizer.pkl: The tokenizer object used to preprocess text during prediction.

- **README.md**: This documentation file.

---

## 🚀 HOW TO RUN ==>

- Open cmd and run following commands ->

1. cd "D:\data d\B. Tech\Projects\FRAUD_DETECTION_SYSTEM"         # To Change Current Directory
2. D:
3. pip install -r requirements.txt                                # To install Dependencies                  
4. python train_model.py                                          # To Train the Model
5. python predict.py                                              # To Run Predictions

---

## ✨ SAMPLE OUTPUT ==>

📩 Message: Win a brand new iPhone for just $1. Claim now<br>
🔍 Prediction: Spam (99.98% confidence)

📩 Message: I will meet you in office.<br>
🔍 Prediction: Not Spam (99.97% confidence)

---

## 📬 CONTACT ==>

For questions or feedback, feel free to reach out!


---









