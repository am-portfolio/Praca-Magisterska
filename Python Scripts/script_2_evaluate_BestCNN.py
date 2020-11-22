from pymodules.mlevaluator import *
from pymodules.anntrainer import *
from pymodules.dnnmaker import*
import tensorflow as tf

# Podstawowa funkcja do tworzenia prostych sieci DNN:
# params: hidden_layers, layer_neurons
def createBestCNN(params, x_shape, num_labels):
    # Regularizer
    regularizer = None
    if(params["l2"] != 0):
        regularizer = tf.keras.regularizers.l2(params["l2"])
        
    # Model sieci
    model = tf.keras.models.Sequential()
    
    # Warstwa wejsciowa
    # I sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        input_shape=(x_shape[0], x_shape[1], 1),
        kernel_size=(5,5), filters=params["filters"][0], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"]))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())
        
    # II sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(5,5), filters=params["filters"][1], activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"]))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())
        
    # III sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(5,5), filters=params["filters"][2], activation='relu'))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())
        
    # Spłaszczanie wejscia
    model.add(tf.keras.layers.Flatten())
    
    # Ukryta warstwa gęsto połączona
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        params["layer_neurons"], activation=tf.nn.relu,
        kernel_regularizer=regularizer, bias_regularizer=None,
    )) 
        
    # Warstwa wyjsciowa (bez softmax)
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        num_labels,
        kernel_regularizer=regularizer, bias_regularizer=None,
    ))
    
    return model








# 4 seria, poszukiwanie najlepszego modelu
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": False,
    "series": 4,

    "model_params": {
        "input_name": ["cqt"],
        
        "filters": [(24, 48, 48)],

        # ZMIANY
        "pool_size": [(2,3), (3,3)],
        "l2": [0.001],
        "dropout": [False],
        "batchnorm": [True],
        "layer_neurons": [64, 128],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [300],
    }
})
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": False,
    "series": 4,

    "model_params": {
        "input_name": ["cqt"],
        
        "filters": [(24, 48, 48)],

        # ZMIANY
        "pool_size": [(2,3), (3,3)],
        "l2": [0.001],
        "dropout": [True],
        "batchnorm": [False],
        "layer_neurons": [64, 128],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [300],
    }
})



# Badanie wpływu metod regularyzacji
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqt"],
        
        "filters": [(24, 48, 48)],
        "pool_size": [(2,3)],
        
        # ZMIANY
        "l2": [0, 0.001],
        "dropout": [True, False],
        "batchnorm": [True, False],
        
        "layer_neurons": [64],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})



# Badanie wpływu liczby neuronów i filtrów
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
    "series": 3,
            
    "model_params": {
        "input_name": ["cqt"],
        
        # ZMIANY 1
        "filters": [(12, 24, 48), (24, 48, 96)],
        
        "pool_size": [(2,3)],
        "l2": [0.001],
        "dropout": [False],
        "batchnorm": [True],
        
        # ZMIANY 2
        "layer_neurons": [64],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
    "series": 3,
            
    "model_params": {
        "input_name": ["cqt"],
        
        # ZMIANY 1
        "filters": [(12, 24, 48), (24, 48, 48), (24, 48, 96)],
        
        "pool_size": [(2,3)],
        "l2": [0.001],
        "dropout": [False],
        "batchnorm": [True],
        
        # ZMIANY 2
        "layer_neurons": [32,128],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})



# Badanie wpływu pool size
evaluateModel({
    "classifier_name": 'BestCNN',

    "create_model_callback": createBestCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
    "series": 2,
        
    "model_params": {
        "input_name": ["cqt"],
        
        "filters": [(24, 48, 48)],
        
        # ZMIANY
        "pool_size": [(3,2), (3,3)],
        
        "l2": [0.001],
        "dropout": [True],
        "batchnorm": [False],
        "layer_neurons": [64],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
