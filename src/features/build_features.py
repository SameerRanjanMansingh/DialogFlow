import pathlib
import pandas as pd
import numpy as np
from preprocess import normalize_text
from feature_transform import tokenizer

max_text_len = 20



def main(df):
    try:
        df['processed_text'] = normalize_text(df['Text'])

        df['count'] = df['Text'].apply(lambda x: len(str(x).split(' ')))

        df = df[df['count'] < max_text_len]
        df = tokenizer(df)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")




def save_data(df, output_path):
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/processed_data.csv', index=False)

if __name__ == '__main__':
    try:
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        
        data_path = home_dir.as_posix() + '/data/raw/friends.csv'
        output_path = home_dir.as_posix() + '/data/processed'


        df = pd.read_csv(data_path)

        df = main(df)
        save_data(df, output_path)
    except Exception as e:
        print(f"An error occurred during execution: {e}")
# data_path = os.path.join("data","processed")

# os.makedirs(data_path)

# train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
# test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"))