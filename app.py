import os
import json
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Load Kaggle credentials
kaggle_credentials = "./assets/kaggle.json"
kaggle_dictionary = json.load(open(kaggle_credentials))

# Set up Kaggle environment variables
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

# Download and unzip dataset (if not already downloaded)
if not os.path.exists("IMDB Dataset.csv"):
    os.system("kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
        zip_ref.extractall()

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Data preprocessing (tokenization, padding, etc.)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["review"])

# Load the pre-trained model
model = load_model('sentiment_analysis_model.h5')

# Prediction function
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

# Streamlit app interface
st.title("Movie Review Sentiment Analysis")
review = st.text_area("Enter your movie review here:")
if st.button("Predict"):
    if review:
        sentiment = predict_sentiment(review)
        st.write(f"The sentiment of the review is: {sentiment}")
