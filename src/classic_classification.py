import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from logger import logging
from data_ingestion import load_data
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import read_yaml
import os

def classic_classification(X_train, X_test, y_train, y_test):

    # Encode labels to consecutive integers
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    le.fit(all_labels)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(np.unique(y_train_enc))

    classifiers = {
        "xgb": XGBClassifier(objective='multi:softmax', use_label_encoder=False, random_state=42),
        "lgb": lgb.LGBMClassifier(
            objective='multiclass',
            num_class=n_classes,
            boosting_type='gbdt',
            max_depth=-1,
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        )
    }

    results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train_enc)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm
        }
        logging.info(f"{name} trained and evaluated.")

    return results

if __name__ == "__main__":
    # Load config
    config = read_yaml("config/urls_config.yaml")
    results_dir = config["results_dir"]["classic_results"]
    df = load_data(config["data_sources"]["product_preprocessed"])
    # Data splits
    lvl1_train_data = np.load(config["nn_model_params"]["lvl1_train_data_path"])
    lvl1_val_data = np.load(config["nn_model_params"]["lvl1_validation_data_path"])
    lvl2_train_data = np.load(config["nn_model_params"]["lvl2_train_data_path"])
    lvl2_val_data = np.load(config["nn_model_params"]["lvl2_validation_data_path"])
    lvl3_train_data = np.load(config["nn_model_params"]["lvl3_train_data_path"])
    lvl3_val_data = np.load(config["nn_model_params"]["lvl3_validation_data_path"])
    # loading data
    X_train_text_1, X_train_num_1, y_train_1 = lvl1_train_data['X_text'], lvl1_train_data['X_num'], lvl1_train_data['y']
    X_val_text_1, X_val_num_1, y_val_1 = lvl1_val_data['X_text'], lvl1_val_data['X_num'],lvl1_val_data['y']
    X_train_text_2, X_train_num_2, y_train_2 = lvl2_train_data['X_text'], lvl2_train_data['X_num'], lvl2_train_data['y']
    X_val_text_2, X_val_num_2, y_val_2 = lvl2_val_data['X_text'], lvl2_val_data['X_num'],lvl2_val_data['y']
    X_train_text_3, X_train_num_3, y_train_3 = lvl3_train_data['X_text'], lvl3_train_data['X_num'], lvl3_train_data['y']
    X_val_text_3, X_val_num_3, y_val_3 = lvl3_val_data['X_text'], lvl3_val_data['X_num'],lvl3_val_data['y']

    # Group related variables into lists for looping
    X_train_texts = [X_train_text_1, X_train_text_2, X_train_text_3]
    X_train_nums  = [X_train_num_1,  X_train_num_2,  X_train_num_3]
    X_val_texts   = [X_val_text_1,   X_val_text_2,   X_val_text_3]
    X_val_nums    = [X_val_num_1,    X_val_num_2,    X_val_num_3]
    y_trains      = [y_train_1,      y_train_2,      y_train_3]
    y_vals        = [y_val_1,        y_val_2,        y_val_3]


    X_combined_test = []
    X_combined_val = []
    y_train_flattened = []
    y_val_flattened = []

    # Loop through each level
    for i in range(3):
        logging.info(f"\nProcessing Level {i+1}...")

        # Convert numeric feature to sparse
        X_length_dense_train = csr_matrix(X_train_nums[i].reshape(-1, 1))
        X_length_dense_val = csr_matrix(X_val_nums[i].reshape(-1, 1))

        # Combine sparse text + numeric feature
        X_combined_train = hstack([X_train_texts[i], X_length_dense_train])
        X_combined_val_matrix = hstack([X_val_texts[i], X_length_dense_val])

        X_combined_test.append(X_combined_train)
        X_combined_val.append(X_combined_val_matrix)

        # Convert one-hot labels → class indices
        y_train_flattened.append(np.argmax(y_trains[i], axis=1))
        y_val_flattened.append(np.argmax(y_vals[i], axis=1))

    for i in range(3):
        logging.info(f"\nRunning Classic Classification for Level {i+1}...")
        result = classic_classification(
            X_combined_test[i], 
            X_combined_val[i],   
            y_train_flattened[i], 
            y_val_flattened[i]
        )

        # 📁 Create a subdirectory per level
        level_dir = os.path.join(results_dir, f"level_{i+1}")
        os.makedirs(level_dir, exist_ok=True)

        # 💾 Save results for each model (xgb, lgb, etc.)
        for model_name, metrics in result.items():
            result_path = os.path.join(level_dir, f"{model_name}_results.txt")
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(f"Level: {i+1}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")

                f.write("📊 Classification Report:\n")
                f.write(str(metrics["classification_report"]) + "\n\n")

                f.write("🧩 Confusion Matrix:\n")
                f.write(str(metrics["confusion_matrix"]) + "\n")

            logging.info(f"Results for {model_name} (Level {i+1}) saved to {result_path}")