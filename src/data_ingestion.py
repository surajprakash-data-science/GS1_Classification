import pandas as pd
import numpy as np
from logger import logging
from sklearn.preprocessing import StandardScaler
from utils import read_yaml

config = read_yaml("config/urls_config.yaml")
file_path = config["data_sources"]["goldstandard_eng"]
lvl1_classification = config["data_params"]["lvl1_label_col"]
lvl2_classification = config["data_params"]["lvl2_label_col"]
lvl3_classification = config["data_params"]["lvl3_label_col"]
product_name = config["data_params"]["product_name"]
product_description = config["data_params"]["product_description"]
product_category = config["data_params"]["product_category"]


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):

    df = df.drop_duplicates().reset_index(drop=True)
    # Remove empty text fields
    df = df[~((df[product_category].str.strip() == "") | (df[product_description].str.strip() == ""))]
    # Combine product text
    df["product"] = (df[product_name].astype(str) + " " +df[product_description].astype(str) + " " +df[product_category].astype(str))
    # Clean product text
    df["product"] = (
        df["product"]
        .str.replace("\n", " ")
        .str.replace("\r", " ")
        .str.replace("\t", " ")
        .str.replace("  ", " ")
        .str.strip()
        .str.lower()
        .str.replace("[^a-zA-Z0-9\s!?., ]", " ", regex=True)
)
    df.dropna(subset=[product_description], inplace=True)
    
    # Feautre engineering
    df["word_count"] = df["product"].apply(lambda x: len(x.split()))
    df["log_word_count"] = np.log1p(df["word_count"])
    scaler = StandardScaler()
    df["standard_log_word_count"] = scaler.fit_transform(df["log_word_count"].values.reshape(-1, 1))

    logging.info("log and standard form of of word count created")
    product_preprocessed = df[["product", lvl1_classification, lvl2_classification, lvl3_classification, "word_count", "log_word_count", "standard_log_word_count"]]
    logging.info("initial preprocessed data is saved")
    product_preprocessed.to_csv("../data/product_preprocessed.csv", index=False)
    return df

if __name__ == "__main__":

    df = load_data(file_path)
    preprocess_data(df)
