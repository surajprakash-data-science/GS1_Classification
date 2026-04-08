import pandas as pd
import os
import yaml
from utils import read_yaml
from sklearn.model_selection import train_test_split


config = read_yaml("config/urls_config.yaml")

train_path = config["data_sources"]["fasttext_train"]
test_path = config["data_sources"]["fasttext_test"]
val_path = config["data_sources"]["fasttext_validation"]
lvl1_classification = config["data_params"]["lvl1_label_col"]
lvl2_classification = config["data_params"]["lvl2_label_col"]
lvl3_classification = config["data_params"]["lvl3_label_col"]
product = config["data_params"]["product"]


from sklearn.utils import resample as sk_resample
import pandas as pd

def conservative_resample(
    df, label_col,
    target_undersample_top5: int = 150,
    target_oversample_cap: int = 100,
    max_oversample_factor: float = 5.0,
    random_state: int = 42
) -> pd.DataFrame:
    category_counts = df[label_col].value_counts()
    top_5_categories = category_counts.nlargest(5).index.tolist()

    balanced_dfs = []

    for category, current_count in category_counts.items():
        category_subset = df[df[label_col] == category]
        resampled_subset = category_subset

        # --- Undersample dominant classes ---
        if category in top_5_categories:
            if current_count > target_undersample_top5:
                resampled_subset = sk_resample(
                    category_subset,
                    replace=False,
                    n_samples=target_undersample_top5,
                    random_state=random_state
                )

        # --- Oversample minority classes ---
        else:
            if current_count < target_oversample_cap:
                target_count = min(
                    int(current_count * max_oversample_factor),
                    target_oversample_cap
                )

                if target_count > current_count:
                    resampled_subset = sk_resample(
                        category_subset,
                        replace=True,
                        n_samples=target_count,
                        random_state=random_state
                    )

        balanced_dfs.append(resampled_subset)

    return pd.concat(balanced_dfs, ignore_index=True)


def preprocessing_fasttext(df):
    df[lvl1_classification] = df[lvl1_classification].str.replace(" ", "_")
    df[lvl2_classification] = df[lvl2_classification].str.replace(" ", "_")
    df[lvl3_classification] = df[lvl3_classification].str.replace(" ", "_")
    #df["lvl2_input"] = df[lvl1_classification].astype(str) + " " + df[product].astype(str)
    #df["lvl3_input"] = (df[lvl1_classification].astype(str) + " " + df[lvl2_classification].astype(str) + " " + df[product].astype(str))

    return df

def fastext_formatting(df, label_col, text_col, column_name):
    # Create a new DataFrame for FastText formatting
    df_formatted = df[[label_col, text_col]].copy()
    df_formatted[label_col] = "__label__" + df_formatted[label_col].astype(str)
    df_formatted[column_name] = df_formatted[label_col] + " " + df_formatted[text_col]
    # splitting for test and validation
    train_full, test_set = train_test_split(df_formatted[column_name], test_size=0.1, random_state=42)
    train_set, val_set = train_test_split(train_full, test_size=0.1, random_state=42)
    
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