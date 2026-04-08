import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from data_ingestion import load_data
from utils import read_yaml

config = read_yaml("config/urls_config.yaml")
df = load_data(config["data_sources"]["product_preprocessed"])
product_col = config["data_params"]["product"]

def create_tokenizers(df, product_col):
    os.makedirs("tokenizers", exist_ok=True)
    num_words = 10000
    oov_token = "<OOV>"

    for level in ["level1", "level2", "level3"]:
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(df[product_col])
        pickle.dump(tokenizer, open(f"tokenizers/tokenizer_{level}.pkl", "wb"))
        print(f"Tokenizer for {level} saved.")
        
create_tokenizers(df, product_col)
