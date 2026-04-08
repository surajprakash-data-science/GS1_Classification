import pandas as pd
import fasttext
import os
from logger import logging
import yaml
from fasttext_preprocessing import conservative_resample
from utils import read_yaml

# Load YAML configuration
config = read_yaml("config/urls_config.yaml")

file_path = config['data_sources']['goldstandard_eng']
model_path = config['models']['models_dir']

lvl1_train = config['output_paths']['lvl1_fasttext_train']
lvl2_train = config["output_paths"]["lvl2_fasttext_train"]
lvl3_train = config["output_paths"]["lvl3_fasttext_train"]

lvl1_val = config['output_paths']['lvl1_fasttext_val']
lvl2_val = config["output_paths"]["lvl2_fasttext_val"]
lvl3_val = config["output_paths"]["lvl3_fasttext_val"]

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
def train_fasttext_model(training_data, val_data, to_model_path):
    model =fasttext.train_supervised(
        input=training_data,
        autotuneValidationFile= val_data
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    return model.save_model(to_model_path)

# FastText Training Script
if __name__ == "__main__":
    from data_ingestion import load_data, preprocess_data
    from fasttext_preprocessing import preprocessing_fasttext, fastext_formatting

    # Load and preprocess data
    df = load_data(file_path)
    #df = conservative_resample(df,label_col="GS1_Level3_Category")
    df = preprocess_data(df)
    df = preprocessing_fasttext(df)
    lvl1_fasttext_data = fastext_formatting(df, 'GS1_Level1_Category', 'product', 'lvl1_fasttext')
    lvl2_fasttext_data = fastext_formatting(df, 'GS1_Level2_Category', 'product', 'lvl2_fasttext')
    lvl3_fasttext_data = fastext_formatting(df, 'GS1_Level3_Category', 'product', 'lvl3_fasttext')

    logging.info("FastText formatting successfull.")

    # Train FastText models
    train_fasttext_model(lvl1_train, lvl1_val, lvl1_model_path)   
    train_fasttext_model(lvl2_train, lvl2_val, lvl2_model_path) 
    train_fasttext_model(lvl3_train, lvl3_val, lvl3_model_path)
    
    logging.info("FastText models trained and saved successfully.")