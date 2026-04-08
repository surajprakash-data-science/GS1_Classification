import os
import pickle
import numpy as np
import fasttext
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODEL_DIR = os.path.join(BASE_DIR, "nn_models")
FASTTEXT_MODEL_DIR = os.path.join(BASE_DIR, "fasttext_models")
TOKENIZER_DIR = os.path.join(SRC_DIR, "tokenizers")

# -------------------------------
# Load Tokenizers
# -------------------------------
with open(os.path.join(TOKENIZER_DIR, "tokenizer_level1.pkl"), "rb") as f:
    tokenizer_lvl1 = pickle.load(f)
with open(os.path.join(TOKENIZER_DIR, "tokenizer_level2.pkl"), "rb") as f:
    tokenizer_lvl2 = pickle.load(f)
with open(os.path.join(TOKENIZER_DIR, "tokenizer_level3.pkl"), "rb") as f:
    tokenizer_lvl3 = pickle.load(f)

# -------------------------------
# Load Label Encoders
# -------------------------------
with open(os.path.join(TOKENIZER_DIR, "label_encoder_level_1.pkl"), "rb") as f:
    le_lvl1 = pickle.load(f)
with open(os.path.join(TOKENIZER_DIR, "label_encoder_level_2.pkl"), "rb") as f:
    le_lvl2 = pickle.load(f)
with open(os.path.join(TOKENIZER_DIR, "label_encoder_level_3.pkl"), "rb") as f:
    le_lvl3 = pickle.load(f)

# -------------------------------
# Load NN Models (Word2Vec + BiLSTM)
# -------------------------------
model_lvl1 = load_model(os.path.join(MODEL_DIR, "lvl1_nn_model_word2vec_weighted.h5"))
model_lvl2 = load_model(os.path.join(MODEL_DIR, "lvl2_nn_model_word2vec_weighted.h5"))
model_lvl3 = load_model(os.path.join(MODEL_DIR, "lvl3_nn_model_word2vec_weighted.h5"))

# -------------------------------
# Load FastText Supervised Models
# -------------------------------
fasttext_lvl1 = fasttext.load_model(os.path.join(FASTTEXT_MODEL_DIR, "fasttext_lvl1_model.bin"))
fasttext_lvl2 = fasttext.load_model(os.path.join(FASTTEXT_MODEL_DIR, "fasttext_lvl2_model.bin"))
fasttext_lvl3 = fasttext.load_model(os.path.join(FASTTEXT_MODEL_DIR, "fasttext_lvl3_model.bin"))

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__, template_folder="templates")

# -------------------------------
# Preprocess Input Text
# -------------------------------
def preprocess_input(text, tokenizer, max_len=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

# -------------------------------
# Predict Category (Neural Network)
# -------------------------------
def predict_nn(text, numeric_input=0):
    X1 = preprocess_input(text, tokenizer_lvl1)
    X2 = preprocess_input(text, tokenizer_lvl2)
    X3 = preprocess_input(text, tokenizer_lvl3)

    num_input = np.array([[numeric_input]])

    pred1 = model_lvl1.predict([X1, num_input])
    pred2 = model_lvl2.predict([X2, num_input])
    pred3 = model_lvl3.predict([X3, num_input])

    lvl1 = le_lvl1.inverse_transform([np.argmax(pred1)])[0]
    lvl2 = le_lvl2.inverse_transform([np.argmax(pred2)])[0]
    lvl3 = le_lvl3.inverse_transform([np.argmax(pred3)])[0]

    return lvl1, lvl2, lvl3

# -------------------------------
# Predict Category (FastText)
# -------------------------------
def predict_fasttext(text):
    lbl1 = fasttext_lvl1.predict(text)[0][0].replace("__label__", "")
    lbl2 = fasttext_lvl2.predict(text)[0][0].replace("__label__", "")
    lbl3 = fasttext_lvl3.predict(text)[0][0].replace("__label__", "")

    return lbl1, lbl2, lbl3

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction_nn = None
    prediction_ft = None

    if request.method == "POST":
        text = request.form.get("product_text", "")
        if text.strip():
            # NN predictions
            nn_lvl1, nn_lvl2, nn_lvl3 = predict_nn(text)
            prediction_nn = {
                "Level 1": nn_lvl1,
                "Level 2": nn_lvl2,
                "Level 3": nn_lvl3
            }

            # FastText predictions
            ft_lvl1, ft_lvl2, ft_lvl3 = predict_fasttext(text)
            prediction_ft = {
                "Level 1": ft_lvl1,
                "Level 2": ft_lvl2,
                "Level 3": ft_lvl3
            }

    return render_template(
        "index.html",
        prediction_nn=prediction_nn,
        prediction_ft=prediction_ft
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
