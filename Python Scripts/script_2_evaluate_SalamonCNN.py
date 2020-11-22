from pymodules.mlevaluator import *
from pymodules.anntrainer import *
from pymodules.dnnmaker import*
import tensorflow as tf

# Podstawowa funkcja do tworzenia prostych sieci DNN:
# params: hidden_layers, layer_neurons
def createSalamonCNN(params, x_shape, num_labels):
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
        kernel_size=(5,5), filters=24, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"]))
    
    # II sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(5,5), filters=48, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"]))
    
    # III sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(5,5), filters=48, activation='relu'))
    
    # Spłaszczanie wejscia
    model.add(tf.keras.layers.Flatten())
    
    # Ukryta warstwa gęsto połączona
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        64, activation=tf.nn.relu,
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





# SalomonCNN z pracy:
# Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification (2017)
evaluateModel({
    "classifier_name": 'SalamonCNN',

    "create_model_callback": createSalamonCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqt", "mels"],
        "row_limit": [None],
        
        "pool_size": [(2,3)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["sgd"],
        "learning_rate": [0.01],
        "batch_size": [100],
        "max_epochs": [250],
    }
})
evaluateModel({
    "classifier_name": 'SalamonCNN',

    "create_model_callback": createSalamonCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [20, 40],
        
        "pool_size": [(1,3)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["sgd"],
        "learning_rate": [0.01],
        "batch_size": [100],
        "max_epochs": [250],
    }
})




evaluateModel({
    "classifier_name": 'SalamonCNN',

    "create_model_callback": createSalamonCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqt", "mels"],
        "row_limit": [None],
        
        "pool_size": [(2,3)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'SalamonCNN',

    "create_model_callback": createSalamonCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [20, 40],
        
        "pool_size": [(1,3)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
