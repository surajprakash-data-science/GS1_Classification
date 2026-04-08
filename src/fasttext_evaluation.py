import fasttext
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_yaml

config = read_yaml("config/urls_config.yaml")
lvl1_model_path = config['models']['lvl1_model']
lvl2_model_path = config['models']['lvl2_model']
lvl3_model_path = config['models']['lvl3_model']

lvl1_val = config['output_paths']['lvl1_fasttext_val']
lvl2_val = config['output_paths']['lvl2_fasttext_val']   
lvl3_val = config['output_paths']['lvl3_fasttext_val']

lvl1_train = config['output_paths']['lvl1_fasttext_train']
lvl2_train = config['output_paths']['lvl2_fasttext_train']   
lvl3_train = config['output_paths']['lvl3_fasttext_train']

def evaluate_fasttext(model_path, test_file, plot_confusion=True):
    # Load model
    model = fasttext.load_model(model_path)

    y_true = []
    y_pred = []

    # Read test data and predict
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            true_label = parts[0].replace("__label__", "")
            text = parts[1] if len(parts) > 1 else ""

            pred_label = model.predict(text, k=1)[0][0].replace("__label__", "")

            y_true.append(true_label)
            y_pred.append(pred_label)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=list(sorted(set(y_true + y_pred))))

    # Output metrics
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Macro-F1     : {macro_f1:.4f}")
    print(f"Weighted-F1  : {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    if plot_confusion:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    return accuracy, macro_f1, weighted_f1, cm

if __name__ == "__main__":
    # import model paths and test data paths from config

    evaluate_fasttext(lvl1_model_path, lvl1_val)
    evaluate_fasttext(lvl1_model_path, lvl1_train)
    evaluate_fasttext(lvl2_model_path, lvl2_val)
    evaluate_fasttext(lvl2_model_path, lvl2_train)
    evaluate_fasttext(lvl3_model_path, lvl3_val)
    evaluate_fasttext(lvl3_model_path, lvl3_train)