import streamlit as st
import joblib

from src.models.predict_model import generate_next_sentence
import pathlib
import sys

from transformers import GPT2Tokenizer


curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
sys.path.append(home_dir.as_posix())

model = joblib.load('models/')
tokenizer = GPT2Tokenizer.from_pretrained("path/to/saved/directory")
vocab_size = tokenizer.vocab_size

st.title("DIALOGFLOW Chat")


col1, col2 = st.columns(2)
with col1:
    user_input = st.text_input("You (Person 1):", "")


with col2:
    max_length = st.slider("Max_length", 1, 20, 5)
    temperature = st.slider0("Temperature", 0, 1, 0.4)

if st.button("Over and out"):
    dilouge = generate_next_sentence(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperature )

    st.write("Person 2:", dilouge)
