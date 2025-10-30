from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import fasttext.util
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from data_ingestion import load_data
import os
import nltk
from logger import logging
from utils import read_yaml

# --------------------------------------------------
# Load Configurations
# --------------------------------------------------
config = read_yaml("config/urls_config.yaml")

fasttext_params = config["fasttext_embedding_params"]
w2v_params = config["word2vec_embedding_params"]
label1 = config["data_params"]["lvl1_label_col"]
label2 = config["data_params"]["lvl2_label_col"]
label3 = config["data_params"]["lvl3_label_col"]
train_path = config["data_sources"]["nn_train"]
test_path = config["data_sources"]["nn_test"]
val_path = config["data_sources"]["nn_validation"]
w2v_model_path = config["models"]["w2v_model"]
product = config["data_params"]["product"]
numeric_col = config["data_params"]["standard_word_count"]

file_path = config["data_sources"]["goldstandard_eng"]

# --------------------------------------------------
# Preprocessing Function
# --------------------------------------------------
def preprocessing_nn(df, product_col, numeric_col, label_col, level):
    np.random.seed(42)
    # Label encode + one-hot target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df[label_col])
    y_onehot = to_categorical(y_encoded)
    logging.info("Label encode + one-hot target done")

    # Tokenization
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df[product_col])
    sequences = tokenizer.texts_to_sequences(df[product_col])
    padded_sequences = pad_sequences(sequences, maxlen=200, padding="post", truncating="post")

    # Numeric input
    X_numeric = df[numeric_col].values.reshape(-1, 1)
    # Train-test-validation split 1
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(padded_sequences, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42)
    X_train_num, X_temp_num = train_test_split(X_numeric, test_size=0.2, random_state=42)
    # Train-test-validation split 1
    X_val_text, X_test_text, y_val, y_test = train_test_split(X_temp_text, y_temp, test_size=0.5, stratify=y_temp,random_state=42)
    X_val_num, X_test_num = train_test_split(X_temp_num, test_size=0.5, random_state=42) # Use the same random state to maintain alignment

    # Save as .npy arrays (more efficient than CSV for NN input)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    np.savez(f"{train_path}_{level}.npy.npz", X_text=X_train_text, X_num=X_train_num, y=y_train)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    np.savez(f"{test_path}_{level}.npy.npz", X_text=X_test_text, X_num=X_test_num, y=y_test)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    np.savez(f"{val_path}_{level}.npy.npz", X_text=X_val_text, X_num=X_val_num, y=y_val)

    logging.info(f"Train/Val/Test for {level} saved as NumPy arrays.")
    return tokenizer

# --------------------------------------------------
# FastText Embeddings
# --------------------------------------------------
def fasttext_embeddings(tokenizer):
    fasttext.util.download_model("en", if_exists="ignore")
    ft = fasttext.load_model('cc.en.300.bin')

    word_index = tokenizer.word_index
    embedding_dim = fasttext_params["embedding_dim"]
    ft_embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim), dtype=np.float32)

    for word, i in word_index.items():
        ft_embedding_matrix[i] = ft.get_word_vector(word)
    logging.info("✅ FastText embeddings created successfully.")
    return ft_embedding_matrix, word_index, embedding_dim

# --------------------------------------------------
# Word2Vec Embeddings
# --------------------------------------------------
def word2vec_training(df, text_col):
    nltk.download('punkt_tab')
    sentences = [word_tokenize(text.lower()) for text in df[text_col].dropna()]

    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=w2v_params["embedding_dim"],
        window=w2v_params["window_size"],
        min_count=w2v_params["min_count"],
        sg=w2v_params["sg"],
        workers=w2v_params["workers"],
        epochs=w2v_params["epoch"]
    )
    os.makedirs(os.path.dirname(w2v_model_path), exist_ok=True)
    w2v_model.save(w2v_model_path)
    logging.info("✅ Word2Vec model trained and saved successfully.")
    return w2v_model

def word2vec_embeddings(tokenizer, w2v_model):
    word_index = tokenizer.word_index
    embedding_dim = w2v_params["embedding_dim"]
    w2v_embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim), dtype=np.float32)
    
    for word, i in word_index.items():
        if word in w2v_model.wv:
            w2v_embedding_matrix[i] = w2v_model.wv[word]
        else:
            w2v_embedding_matrix[i] = np.random.normal(size=(embedding_dim,))
    logging.info("✅ Word2Vec embeddings created successfully.")
    return w2v_embedding_matrix, word_index, embedding_dim

if __name__ == "__main__":

    # Load and preprocess data
    df = load_data(config["data_sources"]["product_preprocessed"])
    preprocessing_nn(df, product, numeric_col, label1, "level_1")
    preprocessing_nn(df, product, numeric_col, label2, "level_2")
    preprocessing_nn(df, product, numeric_col, label3, "level_3")

    w2v_training = word2vec_training(df, product)