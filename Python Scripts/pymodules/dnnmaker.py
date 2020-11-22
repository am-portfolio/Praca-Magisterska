# -*- coding: utf-8 -*-

import tensorflow as tf

# Podstawowa funkcja do tworzenia prostych sieci DNN:
# params: hidden_layers, layer_neurons
def createBasicDNN(params, x_shape, num_labels):
    # Regularizer
    regularizer = None
    if(params["l2"] != 0):
        regularizer = tf.keras.regularizers.l2(params["l2"])
        
    # Model sieci
    model = tf.keras.models.Sequential()
    # Warstwa wejsciowa
    model.add(tf.keras.layers.Flatten(input_shape=(x_shape[0], x_shape[1])))
    
    # Warstwy ukryte
    for l in range(params["hidden_layers"]):
        model.add(tf.keras.layers.Dense(
            params["layer_neurons"], activation=tf.nn.relu,
            # Regularizer tylko dla macierzy W a nie b
            kernel_regularizer=regularizer, bias_regularizer=None,
            # Defaultowa inicializacja tensorflow
            kernel_initializer='glorot_uniform', bias_initializer='zeros',
        )) 
        if(params["dropout"] == True):
            model.add(tf.keras.layers.Dropout(0.5))
        
    # Warstwa wyjsciowa - bez softmax activation=tf.nn.softmax dla lepszej dokładnosci (trzeba potem)
    model.add(tf.keras.layers.Dense(num_labels))
    
    return model