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

FRAUD_DETECTION_SYSTEM/ 
├── data/ 
│   └── sms_spam.csv              # Dataset with labeled SMS messages 
├── model/ 
│   ├── spam_detector_model.h5    # Trained LSTM model 
│   └── tokenizer.pkl             # Fitted tokenizer for preprocessing 
├── preprocessing.py              # Text cleaning and preprocessing functions 
├── train_model.py                # Code to train the LSTM model 
├── predict.py                    # Script to make predictions using trained model 
└── README.md                     # Project documentation

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

📩 Message: Win a brand new iPhone for just $1. Claim now
🔍 Prediction: Spam (99.98% confidence)

📩 Message: I will meet you in office.
🔍 Prediction: Not Spam (99.97% confidence)

---

## 📬 CONTACT ==>

For questions or feedback, feel free to reach out!


---
