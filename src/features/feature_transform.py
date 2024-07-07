from transformers import GPT2Tokenizer
import os
import logging
import pathlib
import sys
import pathlib
import ast

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

from src.data.my_logging_module import setup_custom_logger

logger = setup_custom_logger("my_app", log_level=logging.INFO, log_file="model.log")


def tokenizer(df):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        df['Tokenized_Text'] = df['processed_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

        tokenizer_path = os.path.join("model","tokenizer_path")
        # os.makedirs(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)

        
        return df
    except Exception as e:
        logger.critical("An error occurred while tokenize the data: {e}")
        print(f"Error: An unexpected error occurred while tokenize the data.{e}")
        raise