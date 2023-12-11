from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from text_processing import message_cleaning

# Rest of your code remains the same...


app = Flask(__name__)

# Load the trained model and vectorizer
with open('Naive_bayes_model.pkl', 'rb') as model_file:
    NB_classifier = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to predict spam
def predict_spam(subject_line):
    subject_count = vectorizer.transform([subject_line])
    prediction = NB_classifier.predict(subject_count)
    if prediction[0] == 1:
        return "This is a spam email."
    else:
        return "This is not a spam email."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    result = predict_spam(email_text)
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
