import pandas as pd
import fasttext
import os
import yaml
from utils import read_yaml

# Load YAML configuration
config = read_yaml("../config/urls_config.yaml")

file_path = config['data_sources']['goldstandard_eng']
model_path = config['models']['models_dir']

lvl1_train = config['output_paths']['lvl1_fasttext_train']
lvl2_train = config["output_paths"]["lvl2_fasttext_train"]
lvl3_train = config["output_paths"]["lvl3_fasttext_train"]

# Extract model paths (for saving later)
lvl1_model_path = config["models"]["lvl1_model"]
lvl2_model_path = config["models"]["lvl2_model"]
lvl3_model_path = config["models"]["lvl3_model"]

# Extract training parameters
EPOCHS = config["fasttext_training_params"]["epoch"]
LR = config["fasttext_training_params"]["learning_rate"]
WORD_NGRAMS = config["fasttext_training_params"]["word_ngrams"]
LOSS = config["fasttext_training_params"]["loss_function"]
VERBOSE = config["fasttext_training_params"]["verbose"]


# Function to train FastText model
def train_fasttext_model(training_data, to_model_path):
    model =fasttext.train_supervised(
        input=training_data,
        epoch=EPOCHS,
        lr=LR,
        wordNgrams=WORD_NGRAMS,
        verbose=VERBOSE,
        loss= LOSS
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return model.save_model(to_model_path)

# FastText Training Script
if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data
    from fasttext_preprocessing import preprocessing_fasttext, fastext_formatting

    # Load and preprocess data
    df = load_data(file_path)
    df = preprocess_data(df)
    df = preprocessing_fasttext(df)
    lvl1_fasttext_data = fastext_formatting(df, 'GS1_Level1_Category', 'text', 'lvl1_fasttext')
    lvl2_fasttext_data = fastext_formatting(df, 'GS1_Level2_Category', 'lvl2_input', 'lvl2_fasttext')
    lvl3_fasttext_data = fastext_formatting(df, 'GS1_Level3_Category', 'lvl3_input', 'lvl3_fasttext')

    print("FastText formatting successfull.")

    # Train FastText models
    train_fasttext_model(lvl1_train, lvl1_model_path)   
    train_fasttext_model(lvl2_train, lvl2_model_path)
    train_fasttext_model(lvl3_train, lvl3_model_path)
    
    print("FastText models trained and saved successfully.")