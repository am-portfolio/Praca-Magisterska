import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from timeit import default_timer as timer
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *




# Funkcja zwracająca model kNN o zadanych hiperparametrach:
def createKNN(params, x_shape, num_labels):
    from sklearn.neighbors import KNeighborsClassifier
    kNN = KNeighborsClassifier(
        n_neighbors=params["k"], weights='uniform', algorithm='brute',
    )
    return kNN

# Funkcja trenująca i oceniająca podany model kNN:
def trainAndScoreKNN(kNN, x_train, y_train, x_test, y_test, *args):
    # Główne wyniki:
    scores = {
        "loss": [np.nan],
        "train_loss": [np.nan],
        "accuracy": [],
        "train_accuracy": [np.nan],
        "predictions": [],
        "train_time": [],
        "test_time": [],
    }
    
    print('kNN TRAININ...')
    start = timer()
    kNN.fit(x_train, y_train)
    end = timer()
    scores["train_time"].append(end - start)
    
    print('kNN EVALUATION...')
    start = timer()
    predictions = kNN.predict_proba(x_test)
    end = timer()
    scores["test_time"].append((end - start)/x_test.shape[0])
    scores["predictions"].append(predictions)

    y_pred = np.argmax(predictions, axis=1)
    scores["accuracy"].append(sklearn.metrics.accuracy_score(y_test, y_pred))
    
    return scores





#---------------------------------------------------#
#---------------------------------------------------#

evaluateModel({
    "classifier_name": 'kNN',

    "create_model_callback": createKNN,
    "train_and_score_model_callback": trainAndScoreKNN,
    
    "flatten": True,
        
    "model_params": {
        "input_name": ["mfcc", "cqcc", "mfccl2", "mfccmax", "mels", "cqt"],
        "k": np.arange(1,21,2),
        "row_limit": [None, 80, 60, 40, 30, 25, 20, 15],
    }
})