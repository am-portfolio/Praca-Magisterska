from pymodules.mlevaluator import *
from pymodules.anntrainer import *
from pymodules.dnnmaker import*
import tensorflow as tf

# Podstawowa funkcja do tworzenia prostych sieci DNN:
# params: hidden_layers, layer_neurons
def createHuzaifahCNN(params, x_shape, num_labels):
    # Regularizer
    regularizer = None
    if(params["l2"] != 0):
        regularizer = tf.keras.regularizers.l2(params["l2"])
        
    # Model sieci
    model = tf.keras.models.Sequential()

    # Zmniejszanie wymiarowości uśrednienie
    model.add(tf.keras.layers.AveragePooling2D(
      input_shape=(x_shape[0], x_shape[1], 1),
      pool_size=params["pool_size"],
    ))

    # Warstwa wejsciowa
    # I sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(3,3), filters=180, activation='relu',
        kernel_regularizer=regularizer, bias_regularizer=None))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(4,4)))
    
    # Spłaszczanie wejscia
    model.add(tf.keras.layers.Flatten())
    
    # Ukryta warstwa gęsto połączona
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        800, activation=None,
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




# HuzaifahCNN z pracy:
# Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks
evaluateModel({
    "classifier_name": 'HuzaifahCNN',

    "create_model_callback": createHuzaifahCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,
        
    "model_params": {
        "input_name": ["cqt", "mels"],
        "row_limit": [None],
        
        "pool_size": [(1,4)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'HuzaifahCNN',

    "create_model_callback": createHuzaifahCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,
        
    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [40, 20],
        
        "pool_size": [(1,2)],

        "l2": [0.001],
        "dropout": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})