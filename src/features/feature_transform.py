from transformers import GPT2Tokenizer
import os

def tokenizer(df):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        df['Tokenized_Text'] = df['processed_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        tokenizer_path = os.path.join("model","tokenizer_path")
        os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)

        return df
    except Exception as e:
        print(f"Error: An unexpected error occurred while tokenize the data.")
        print(e)
        raise