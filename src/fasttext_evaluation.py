import fasttext
from utils import read_yaml

config = read_yaml("../config/urls_config.yaml")
lvl1_model_path = config['models']['lvl1_model']
lvl2_model_path = config['models']['lvl2_model']
lvl3_model_path = config['models']['lvl3_model']

lvl1_test = config['output_paths']['lvl1_fasttext_test']
lvl2_test = config['output_paths']['lvl2_fasttext_test']   
lvl3_test = config['output_paths']['lvl3_fasttext_test']

lvl1_train = config['output_paths']['lvl1_fasttext_train']
lvl2_train = config['output_paths']['lvl2_fasttext_train']   
lvl3_train = config['output_paths']['lvl3_fasttext_train']

def evaluate_fasttext_model(model_path, test_data_path):
    
    model = fasttext.load_model(model_path)
    result = model.test(test_data_path)
    total_samples = result[0]
    precision = result[1]
    recall = result[2]
    print(f"Total samples: {total_samples}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print('\n')
    return precision, recall


if __name__ == "__main__":
    # import model paths and test data paths from config
    evaluate_fasttext_model(lvl1_model_path, lvl1_test)
    evaluate_fasttext_model(lvl1_model_path, lvl1_train)
    evaluate_fasttext_model(lvl2_model_path, lvl2_test)
    evaluate_fasttext_model(lvl2_model_path, lvl2_train)
    evaluate_fasttext_model(lvl3_model_path, lvl3_test)
    evaluate_fasttext_model(lvl3_model_path, lvl3_train)
