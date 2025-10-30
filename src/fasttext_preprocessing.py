import pandas as pd
import os
import yaml
from utils import read_yaml
from sklearn.model_selection import train_test_split


config = read_yaml("../config/urls_config.yaml")

train_path = config["data_sources"]["fasttext_train"]
test_path = config["data_sources"]["fasttext_test"]
val_path = config["data_sources"]["fasttext_validation"]
lvl1_classification = config["data_params"]["lvl1_label_col"]
lvl2_classification = config["data_params"]["lvl2_label_col"]
lvl3_classification = config["data_params"]["lvl3_label_col"]
product = config["data_params"]["product"]

def preprocessing_fasttext(df):
    df["lvl2_input"] = lvl1_classification + " " + product
    df["lvl3_input"] = lvl1_classification + " " + lvl2_classification + " " + product

    return df

def fastext_formatting(df, label_col, text_col, column_name):
    # Create a new DataFrame for FastText formatting
    df_formatted = df[[label_col, text_col]].copy()
    df_formatted[label_col] = "__label__" + df_formatted[label_col].astype(str)
    df_formatted[column_name] = df_formatted[label_col] + " " + df_formatted[text_col]
    # splitting for test and validation
    train_set, test_set = train_test_split(df_formatted[column_name], test_size=0.1, random_state=42)
    train_set, val_set = train_test_split(df_formatted[column_name], test_size=0.1, random_state=42 )
    
    # Save the training datasets to files
    os.makedirs(train_path, exist_ok=True)
    train_data_path = os.path.join(train_path, f"{column_name}_train.txt")
    train_set.to_csv(train_data_path, index=False, header=False)
    # Save the test datasets to files
    os.makedirs(test_path, exist_ok=True)
    test_data_path = os.path.join(test_path, f"{column_name}_test.txt")
    test_set.to_csv(test_data_path, index=False, header=False)
    # Save the validatoin datasets to files
    os.makedirs(val_path, exist_ok=True)
    val_data_path = os.path.join(val_path, f"{column_name}_val.txt")
    val_set.to_csv(val_data_path, index=False, header=False)
    
    return train_data_path, test_data_path, val_data_path