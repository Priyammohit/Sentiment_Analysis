
import os
import json
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Load Kaggle credentials
kaggle_credentials = "./assets/kaggle.json"
kaggle_dictionary = json.load(open(kaggle_credentials))

# Set up Kaggle environment variables
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

# Download and unzip dataset
os.system("kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
    zip_ref.extractall()

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")

# Data preprocessing (tokenization, padding, etc.)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["review"])
X = pad_sequences(tokenizer.texts_to_sequences(data["review"]), maxlen=200)
Y = data["sentiment"].replace({"positive": 1, "negative": 0})

# Model definition
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=5, batch_size=64, validation_split=0.2)

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
