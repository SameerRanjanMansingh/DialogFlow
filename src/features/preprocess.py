import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import numpy as np


nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df=df.apply(lambda content : lower_case(content))
    df=df.apply(lambda content : remove_stop_words(content))
    df=df.apply(lambda content : removing_numbers(content))
    df=df.apply(lambda content : removing_punctuations(content))
    df=df.apply(lambda content : removing_urls(content))
    df=df.apply(lambda content : lemmatization(content))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= removing_numbers(sentence)
    sentence= removing_punctuations(sentence)
    sentence= removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence




def split_data(df):
    sequence_length = 5  # Reduced sequence length for demonstration

    x = []
    y = []

    try:
        for _, row in df.iterrows():
            tokens = row['Tokenized_Text']
            if len(tokens) > sequence_length:  # Ensure there are enough tokens for a sequence
                for i in range(len(tokens) - sequence_length):
                    x.append(tokens[i:i+sequence_length])  # Add sequence of previous dialogues to X
                    y.append(tokens[i+sequence_length])    # Add next dialogue token to y

        # Convert X and y to numpy arrays
        x = np.array(x)
        y = np.array(y)

        return x, y
    
    except Exception as e:
        print(f"An error occurred: {e}")
    