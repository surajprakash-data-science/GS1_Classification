from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import read_yaml
import config
config = read_yaml("config/urls_config.yaml")

def evaluate_lstm(
    model_path,
    X_test_text,
    X_test_num,
    y_test_onehot,
    plot_confusion=True):
    # Load model
    model = load_model(model_path)

    # Predict probabilities
    y_pred_probs = model.predict([X_test_text, X_test_num], verbose=0)

    # Convert to class indices
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_onehot, axis=1)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print metrics
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}")
    print(f"Weighted-F1  : {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

   # Optional plot
    if plot_confusion:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap="Blues", annot=False)
        plt.title("Confusion Matrix (LSTM)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()
    return accuracy, macro_f1, weighted_f1, cm

if __name__ == "__main__":
    w2vmodel_level_1 = config["models"]["lvl1_nn_model_word2vec_weighted"]
    w2vmodel_level_2 = config["models"]["lvl2_nn_model_word2vec_weighted"]
    w2vmodel_level_3 = config["models"]["lvl3_nn_model_word2vec_weighted"]
    ftmodel_level_1 = config["models"]["lvl1_nn_model_fasttext"]
    ftmodel_level_2 = config["models"]["lvl2_nn_model_fasttext"]
    ftmodel_level_3 = config["models"]["lvl3_nn_model_fasttext"]

    lvl1_test_data = np.load(config["nn_model_params"]["lvl1_test_data_path"])
    X_test_text_1, X_test_num_1, y_test_1 = lvl1_test_data['X_text'], lvl1_test_data['X_num'], lvl1_test_data['y']
    lvl2_test_data = np.load(config["nn_model_params"]["lvl2_test_data_path"])
    X_test_text_2, X_test_num_2, y_test_2 = lvl2_test_data['X_text'], lvl2_test_data['X_num'], lvl2_test_data['y']
    lvl3_test_data = np.load(config["nn_model_params"]["lvl3_test_data_path"])
    X_test_text_3, X_test_num_3, y_test_3 = lvl3_test_data['X_text'], lvl3_test_data['X_num'], lvl3_test_data['y']

    print("Evaluating Level 1 Word2Vec-based LSTM Model:")
    evaluate_lstm(
        model_path=w2vmodel_level_1,
        X_test_text=X_test_text_1,
        X_test_num=X_test_num_1,
        y_test_onehot=y_test_1,
        plot_confusion=True
    )
    print("Evaluating Level 2 Word2Vec-based LSTM Model:")
    evaluate_lstm(
        model_path=w2vmodel_level_2,
        X_test_text=X_test_text_2,
        X_test_num=X_test_num_2,
        y_test_onehot=y_test_2,
        plot_confusion=True
    )
    print("Evaluating Level 3 Word2Vec-based LSTM Model:")
    evaluate_lstm(
        model_path=w2vmodel_level_3,
        X_test_text=X_test_text_3,
        X_test_num=X_test_num_3,
        y_test_onehot=y_test_3,
        plot_confusion=True
    )
    print("Evaluating Level 1 FastText-based LSTM Model:")
    evaluate_lstm(
        model_path=ftmodel_level_1,
        X_test_text=X_test_text_1,
        X_test_num=X_test_num_1,
        y_test_onehot=y_test_1,
        plot_confusion=True
    )
    print("Evaluating Level 2 FastText-based LSTM Model:")
    evaluate_lstm(
        model_path=ftmodel_level_2,
        X_test_text=X_test_text_2,
        X_test_num=X_test_num_2,
        y_test_onehot=y_test_2,
        plot_confusion=True
    )
    print("Evaluating Level 3 FastText-based LSTM Model:")
    evaluate_lstm(
        model_path=ftmodel_level_3,
        X_test_text=X_test_text_3,
        X_test_num=X_test_num_3,
        y_test_onehot=y_test_3,
        plot_confusion=True
    )