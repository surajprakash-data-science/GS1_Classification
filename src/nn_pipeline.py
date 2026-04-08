from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, Input, Concatenate
from tensorflow.keras import regularizers, optimizers
from utils import read_yaml
# all necessary functions are imported from data_preprocessing
from data_preprocessing import preprocessing_nn, fasttext_embeddings, word2vec_embeddings, get_class_weights 
from gensim.models import Word2Vec
from logger import logging
import numpy as np
from data_ingestion import load_data

# Load config and parameters
config = read_yaml("config/urls_config.yaml")
nn_params = config["nn_model_params"]
w2v_model_path = config["models"]["w2v_model"]
label1 = config["data_params"]["lvl1_label_col"]
label2 = config["data_params"]["lvl2_label_col"]
label3 = config["data_params"]["lvl3_label_col"]
product = config["data_params"]["product"]
softmax_activation = config["nn_model_params"]["softmax_activation"]
relu_activation = config["nn_model_params"]["relu_activation"]
regularizer = config["nn_model_params"]["regularizer"]
learning_rate = config["nn_model_params"]["learning_rate"]
numeric_col = config["data_params"]["standard_word_count"]
WEIGHTED_MODEL_PATH_1 = config["models"]["lvl1_nn_model_word2vec"].replace(".h5", "_weighted.h5")
WEIGHTED_MODEL_PATH_2 = config["models"]["lvl2_nn_model_word2vec"].replace(".h5", "_weighted.h5")
WEIGHTED_MODEL_PATH_3 = config["models"]["lvl3_nn_model_word2vec"].replace(".h5", "_weighted.h5")

# LSTM model (Functional API for Dual Input)
def build_nn_model(X_train_text, y_train, embedding_matrix, word_index, embedding_dim):
    # Text Input
    text_input = Input(shape=(X_train_text.shape[1],),name="text_input")
    x = Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=X_train_text.shape[1], trainable=False)(text_input)

    x = Bidirectional(LSTM(128, return_sequences=True,  kernel_regularizer=regularizers.l2(regularizer)))(x)
    x = Dropout(0.4)(x)

    text_vector = GlobalMaxPooling1D()(x)
    # Numeric Input
    numeric_input = Input(shape=(1,), name="word_count")
    combined = Concatenate()([text_vector, numeric_input])

    y = combined
    output = Dense(y_train.shape[1], activation=softmax_activation)(y)
    model = Model(inputs=[text_input, numeric_input], outputs=output)

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=nn_params['loss'], metrics=nn_params['metrics'])
    return model


if __name__ == "__main__":
    df = load_data(config["data_sources"]["product_preprocessed"])
    
    # LVL1 TRAINING MODELS ================================================================================================================
    # Preprocess for NN
    tokenizer = preprocessing_nn(df, product, numeric_col, label1, "level_1")

    # LOAD the saved data arrays for the model training
    lvl1_train_data = np.load(config["nn_model_params"]["lvl1_train_data_path"])
    lvl1_val_data = np.load(config["nn_model_params"]["lvl1_validation_data_path"])
    
    # Load feature and label arrays
    X_train_text_1, X_train_num_1, y_train_1 = lvl1_train_data['X_text'], lvl1_train_data['X_num'], lvl1_train_data['y']
    X_val_text_1, X_val_num_1, y_val_1 = lvl1_val_data['X_text'], lvl1_val_data['X_num'],lvl1_val_data['y']

    # Calculate Class Weights for the Weighted Experiment
    class_weights_1 = get_class_weights(y_train_1) 
    
    '''    # Load Embeddings
    w2v_model_load = Word2Vec.load(w2v_model_path)
    w2v_embedding_matrix, word_index, embedding_dim = word2vec_embeddings(tokenizer, w2v_model_load)'''
    
    
    #  FastText Embeddings Model   
    ft_embedding_matrix, word_index, embedding_dim = fasttext_embeddings(tokenizer)
    model_ft = build_nn_model(X_train_text_1, y_train_1, ft_embedding_matrix, word_index, embedding_dim)

    ft_history = model_ft.fit( x=[X_train_text_1, X_train_num_1], 
        y=y_train_1, 
        epochs=nn_params['epochs'], 
        batch_size=nn_params['batch_size'], 
        validation_data=([X_val_text_1, X_val_num_1], y_val_1),
        # CRITICAL: Apply the calculated class weights
        class_weight=class_weights_1)
    model_ft.save(config["models"]["lvl1_nn_model_fasttext"])
    print(f"FastText-based model saved at {config['models']['lvl1_nn_model_fasttext']}") 
    
    '''
    ## --- EXPERIMENT 1A: BASELINE MODEL (NO CLASS WEIGHTS) ---
    logging.info("Starting LVL1 BASELINE Model Training...")
    w2v_model_baseline = build_nn_model(X_train_text_1, y_train_1, w2v_embedding_matrix, word_index, embedding_dim)
    w2v_history_baseline = w2v_model_baseline.fit(
        x=[X_train_text_1, X_train_num_1], 
        y=y_train_1, 
        epochs=nn_params['epochs'], 
        batch_size=nn_params['batch_size'], 
        validation_data=([X_val_text_1, X_val_num_1], y_val_1)
    )
    w2v_model_baseline.save(config["models"]["lvl1_nn_model_word2vec"]) # Save to the default path
    logging.info("Level 1 BASELINE Word2Vec-based model saved")
    '''
    
    '''    ## --- EXPERIMENT 1B: WEIGHTED MODEL (CLASS WEIGHTS) ---
    logging.info("Starting LVL1 WEIGHTED Model Training...")
    w2v_model_weighted_lvl1 = build_nn_model(X_train_text_1, y_train_1, w2v_embedding_matrix, word_index, embedding_dim)
    w2v_history_weighted_lvl1 = w2v_model_weighted_lvl1.fit(
        x=[X_train_text_1, X_train_num_1], 
        y=y_train_1, 
        epochs=nn_params['epochs'], 
        batch_size=nn_params['batch_size'], 
        validation_data=([X_val_text_1, X_val_num_1], y_val_1),
        # CRITICAL: Apply the calculated class weights
        class_weight=class_weights_1
    )
    w2v_model_weighted_lvl1.save(WEIGHTED_MODEL_PATH_1) # Save to the custom weighted path
    logging.info("Level 1 WEIGHTED Word2Vec-based model saved")'''
    
    # LVL2 TRAINING MODELS==============================================================================================================
    # Preprocess for NN
    tokenizer = preprocessing_nn(df, product, numeric_col, label2, "level_2")

    # LOAD the saved data arrays for the model training
    lvl2_train_data = np.load(config["nn_model_params"]["lvl2_train_data_path"])
    lvl2_val_data = np.load(config["nn_model_params"]["lvl2_validation_data_path"])
    
    # Load feature and label arrays
    X_train_text_2, X_train_num_2, y_train_2 = lvl2_train_data['X_text'], lvl2_train_data['X_num'], lvl2_train_data['y']
    X_val_text_2, X_val_num_2, y_val_2 = lvl2_val_data['X_text'], lvl2_val_data['X_num'],lvl2_val_data['y']

    # Calculate Class Weights for the Weighted Experiment
    class_weights_2 = get_class_weights(y_train_2)
    
    
    #  FastText Embeddings Model
    ft_embedding_matrix, word_index, embedding_dim = fasttext_embeddings(tokenizer)
    model_ft = build_nn_model(X_train_text_2, y_train_2, ft_embedding_matrix, word_index, embedding_dim)

    ft_history = model_ft.fit(x=[X_train_text_2, X_train_num_2], y=y_train_2, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_data=([X_val_text_2, X_val_num_2], y_val_2), class_weight=class_weights_2)
    model_ft.save(config["models"]["lvl2_nn_model_fasttext"])
    print(f"FastText-based model saved at {config['models']['lvl2_nn_model_fasttext']}")

    '''    #  Word2Vec Embeddings Model
    w2v_model_load = Word2Vec.load(w2v_model_path)
    w2v_embedding_matrix, word_index, embedding_dim = word2vec_embeddings(tokenizer, w2v_model_load)
    w2v_model_weighted_lvl2 = build_nn_model(X_train_text_2, y_train_2, w2v_embedding_matrix, word_index, embedding_dim)
    w2v_history_weighted_lvl2 = w2v_model_weighted_lvl2.fit(x=[X_train_text_2, X_train_num_2], y=y_train_2, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_data=([X_val_text_2, X_val_num_2], y_val_2), class_weight=class_weights_2)
    w2v_model_weighted_lvl2.save(WEIGHTED_MODEL_PATH_2)
    logging.info("level 2 Word2Vec-based model saved")'''
    

    # LVL3 TRAINING MODELS==================================================================================================================
        # Preprocess for NN
    tokenizer = preprocessing_nn(df, product, numeric_col, label3, "level_3")

    # LOAD the saved data arrays for the model training
    lvl3_train_data = np.load(config["nn_model_params"]["lvl3_train_data_path"])
    lvl3_val_data = np.load(config["nn_model_params"]["lvl3_validation_data_path"])
    
    # Load feature and label arrays
    X_train_text_3, X_train_num_3, y_train_3 = lvl3_train_data['X_text'], lvl3_train_data['X_num'], lvl3_train_data['y']
    X_val_text_3, X_val_num_3, y_val_3 = lvl3_val_data['X_text'], lvl3_val_data['X_num'],lvl3_val_data['y']

    # Calculate Class Weights for the Weighted Experiment
    class_weights_3 = get_class_weights(y_train_3)
    
    #  FastText Embeddings Model
    ft_embedding_matrix, word_index, embedding_dim = fasttext_embeddings(tokenizer)
    model_ft = build_nn_model(X_train_text_3, y_train_3, ft_embedding_matrix, word_index, embedding_dim)

    ft_history = model_ft.fit(x=[X_train_text_3, X_train_num_3], y=y_train_3, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_data=([X_val_text_3, X_val_num_3], y_val_3), class_weight=class_weights_3)
    model_ft.save(config["models"]["lvl3_nn_model_fasttext"])
    print(f"FastText-based model saved at {config['models']['lvl3_nn_model_fasttext']}")
    

    '''    #  Word2Vec Embeddings Model
    w2v_model_load = Word2Vec.load(w2v_model_path)
    w2v_embedding_matrix, word_index, embedding_dim = word2vec_embeddings(tokenizer, w2v_model_load)
    w2v_model_weighted_lvl3 = build_nn_model(X_train_text_3, y_train_3, w2v_embedding_matrix, word_index, embedding_dim)
    w2v_history_weighted_lvl3 = w2v_model_weighted_lvl3.fit(x=[X_train_text_3, X_train_num_3], y=y_train_3, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], validation_data=([X_val_text_3, X_val_num_3], y_val_3), class_weight=class_weights_3)
    w2v_model_weighted_lvl3.save(WEIGHTED_MODEL_PATH_3)
    logging.info("level 3 Word2Vec-based model saved")'''