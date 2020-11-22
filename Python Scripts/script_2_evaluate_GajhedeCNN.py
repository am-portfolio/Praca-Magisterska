from pymodules.mlevaluator import *
from pymodules.anntrainer import *
from pymodules.dnnmaker import*
import tensorflow as tf

# Podstawowa funkcja do tworzenia prostych sieci DNN:
# params: dropout, batchnorm
def createGajhedeCNN(params, x_shape, num_labels):
    # Model sieci
    model = tf.keras.models.Sequential()

    # Zmniejszanie wymiarowości uśrednienie
    model.add(tf.keras.layers.AveragePooling2D(
      input_shape=(x_shape[0], x_shape[1], 1),
      pool_size=(1,10),
    ))
    
    # Warstwa wejsciowa
    # I sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(2,4), filters=16, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"], padding="same"))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())
    
    # II sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(3,3), filters=32, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"], padding="same"))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())
    
    # III sekcja konwolucyjno poolingowa.
    model.add(tf.keras.layers.Conv2D(
        kernel_size=(3,3), filters=64, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=params["pool_size"], padding="same"))
    if params["batchnorm"] == True:
        model.add(tf.keras.layers.BatchNormalization())    
    
    # Spłaszczanie wejscia
    model.add(tf.keras.layers.Flatten())
    
    # Ukryta warstwa gęsto połączona
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(
        256, activation='tanh',
    )) 
        
    # Warstwa wyjsciowa (bez softmax)
    if params["dropout"] == True:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_labels))
    
    return model




# GajhedeCNN z pracy:
# Convolutional Neural Networks with Batch Normalization for Classifying Hi-hat, Snare, and Bass Percussion Sound Samples
evaluateModel({
    "classifier_name": 'GajhedeCNN',

    "create_model_callback": createGajhedeCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqt", "mels"],
        "row_limit": [None],
        
        "pool_size": [(5,1)],

        "dropout": [True],
        "batchnorm": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'GajhedeCNN',

    "create_model_callback": createGajhedeCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [40],
        
        "pool_size": [(3,1)],

        "dropout": [True],
        "batchnorm": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'GajhedeCNN',

    "create_model_callback": createGajhedeCNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,
        
    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [20],
        
        "pool_size": [(2,1)],

        "dropout": [True],
        "batchnorm": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})