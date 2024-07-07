import streamlit as st
import joblib

from src.models.predict_model import generate_next_sentence
from transformers import GPT2Tokenizer




model = joblib.load('models/model.joblib')
tokenizer = GPT2Tokenizer.from_pretrained("model/tokenizer_path")
vocab_size = tokenizer.vocab_size

st.title("DIALOGFLOW Chat")


col1, col2 = st.columns(2)
with col1:
    user_input = st.text_area("You (Person 1):", value="Hey,", height=130)


with col2:
    max_length = st.slider("Max_length", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4)

if st.button("Over and out"):
    dilouge = generate_next_sentence(
        model=model,
        tokenizer=tokenizer,
        text_sequence=user_input,
        max_length=max_length,
        temperature=temperature )

    st.write("Person 2:", dilouge)
