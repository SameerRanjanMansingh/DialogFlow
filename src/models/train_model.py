import sys
import logging
import joblib
import pandas as pd
import yaml
import pathlib

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

from src.features.preprocess import split_data
from src.data.my_logging_module import setup_custom_logger


from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer


logger = setup_custom_logger("my_app", log_level=logging.INFO, log_file="model.log")


tokenizer = GPT2Tokenizer.from_pretrained('model/tokenizer_path')


curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
sys.path.append(home_dir.as_posix())


sequence_length = yaml.safe_load(open('params.yaml', 'r'))['train_model']['sequence_length']
embedding_dim = yaml.safe_load(open('params.yaml', 'r'))['train_model']['embedding_dim']


vocab_size = tokenizer.vocab_size
lstm_units = 200



def build_model():
    try:
        input_text = Input(shape=(sequence_length,))

        embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_text)

        lstm_outputs, _, _ = LSTM(lstm_units, return_state=True, return_sequences=False, dropout=0.4, recurrent_dropout=0.4)(embedding_layer)
        dense_output = Dense(vocab_size, activation='softmax')(lstm_outputs)

        model = Model(inputs=input_text, outputs=dense_output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        logger.info("Building the model")

        return model
    except Exception as e:
        print(f"An error occurred while building the model: {e}")
        logger.error(f"An error occurred while building the model: {e}")
        return None



def save_model(model, output_path):
    try:
        joblib.dump(model, output_path + "/model.joblib")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")


def main():
    input_file = '/data/processed'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + "/models"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    try:
        df=pd.read_csv(data_path+ '/processed_data.csv')
        X, y = split_data(df=df)

        print(X.shape)
        print(y.shape)


        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

        model = build_model()
        if model:
            callback = EarlyStopping(monitor='loss', patience=2)

            model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1, 
            batch_size=128, 
            verbose=1,
            callbacks=[callback]
            )
        
            save_model(model=model, output_path=output_path)

            
            logger.info("Training completed")
    except Exception as e:
        logger.critical(f"An error occurred during training: {e}")
        print(f"An error occurred during training: {e}")



if __name__ == "__main__":
    main()
