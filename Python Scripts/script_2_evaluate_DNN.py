from pymodules.mlevaluator import *
from pymodules.anntrainer import *
from pymodules.dnnmaker import*


# SERIA 1
evaluateModel({
    "classifier_name": 'DNN',
    "series": 1,

    "create_model_callback": createBasicDNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "model_params": {
        "input_name": ["cqcc", "cqt", "mfcc", "mels"],
        "row_limit": [30],
        
        "hidden_layers": [1,2,3],
        "layer_neurons": [128, 256, 512, 1024],
        "l2": [0],
        "dropout": [False],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [50],
    }
})


# SERIA 2
evaluateModel({
    "classifier_name": 'DNN',
    "series": 2,

    "create_model_callback": createBasicDNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [10, 15, 20, 25, 40, 60, 80],
        
        "hidden_layers": [2],
        "layer_neurons": [256, 512, 1024],
        "l2": [0],
        "dropout": [False],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [50],
    }
})


# SERIA 3
evaluateModel({
    "classifier_name": 'DNN',
    "series": 3,

    "create_model_callback": createBasicDNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,

    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [20, 40],
        
        "hidden_layers": [2],
        "layer_neurons": [512, 1024],
        "l2": [0, 0.001],
        "dropout": [True],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})
evaluateModel({
    "classifier_name": 'DNN',
    "series": 3,

    "create_model_callback": createBasicDNN,
    "train_and_score_model_callback": trainAndScoreNeuralNetwork,

    "save_models": True,

    "model_params": {
        "input_name": ["cqcc", "mfcc"],
        "row_limit": [20, 40],
        
        "hidden_layers": [2],
        "layer_neurons": [512, 1024],
        "l2": [0.001],
        "dropout": [False],

        "optimizer": ["adam"],
        "learning_rate": [0.001],
        "batch_size": [100],
        "max_epochs": [150],
    }
})