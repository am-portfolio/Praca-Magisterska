# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import sklearn
import sklearn.metrics
from timeit import default_timer as timer
from .utilities import *


# Funkcja trenująca i oceniająca podany model kNN:
def trainAndScoreNeuralNetwork(model, x_train, y_train, x_test, y_test, model_params, scores = None, verbose = 0):
    
    # Funkcja straty to zawsze crossentropia
    lossFunction = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Gdy model jest uczony pierwszy raz...
    if scores == None:
        # Inicializacja scores:
        scores = {
            "loss": [],
            "train_loss": [],
            "accuracy": [],
            "train_accuracy": [],
            "predictions": [],
            "train_time": [],
            "test_time": [],
        }
        # Parametry uczenia i kompilacja:
        learning_rate = model_params["learning_rate"]
        optimizer  = None
        if (model_params["optimizer"] == "adam"):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif (model_params["optimizer"] == "sgd"):
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=lossFunction, metrics=['accuracy'])
    
    # Parametry:
    batch_size = model_params["batch_size"]
    if len(scores["test_time"]) == 0:
        # Model jest uczony od zera
        max_epochs = model_params["max_epochs"]
    else:
        # Tylko pozostała częsc epok
        max_epochs = model_params["max_epochs"] - (len(scores["test_time"]) - 1)


    # Wyznacza predykcje dla x_test, mieryz czas, wyznacza stratę
    # i dokładnosc a potem dopsuje do scores.
    def evaluateTestSet():
        start = timer()
        logits = model.predict(x_test)
        predictions = tf.nn.softmax(logits).numpy()
        end = timer()
        scores["predictions"].append(predictions)
        scores["test_time"].append((end - start)/x_test.shape[0])      
        
        # Wyznaczenie straty (nie uwzględnia regularyzacji)
        loss_v = lossFunction(y_test, logits).numpy()
        scores['loss'].append(loss_v)
        
        # Wyznaczenie dokładnosci (na całym modelu bez dropout)
        y_pred = np.argmax(predictions, axis=1)
        accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
        scores['accuracy'].append(accuracy)
    
    # Wyznaczenie dokładnosci i straty startowej przed uczeniem.
    # Tylko gdy model jest uczony pierwszy raz (puste tablice)
    if len(scores["train_time"]) == 0:
        scores["train_time"].append(0)
        if verbose > 0:
            print('ANN PRE-TRAINING EVALUATION...')
        # Przed uczeniem, zbiór uczący
        test_loss, test_acc = model.evaluate(x_train, y_train, verbose=0)
        scores['train_loss'].append(test_loss)
        scores['train_accuracy'].append(test_acc)
        # Przed uczeniem, zbiór testowy
        evaluateTestSet()
    


    # Uczenie i walidowanie modelu.
    class PredictionCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs):
            if verbose > 0:
                print(f'ANN TRAINING - EPOCH {epoch+1}/{max_epochs}...')
            self.start_epoch = timer()
   
        # epoch - numer poprzedniej epoki: tz. 0, 1, 2...
        # logs - {'loss': _, 'accuracy': _, 'val_loss': _, 'val_accuracy': _}
        def on_epoch_end(self, epoch, logs):
            # Zapisanie czasu trenowania.
            end_epoch = timer()
            scores["train_time"].append(scores["train_time"][-1] + end_epoch - self.start_epoch)

            # Wykonanie predykcji dla zestawu testowego (na całym modelu bez dropoutu)
            evaluateTestSet()
            
            # Strata uwzględniająca regularyzację i dropout liczona w W TRAKCIE 
            # epok a nie na końcu. Wynik bedzie niższy niż gdyby go ręcznie tu wyliczyc.
            scores['train_loss'].append(logs['loss'])
            # Średnia dokładnosci predykcji usredniana W TRAKCIE epoki z modelami z dropout.
            # Wynik bedzie niższy niż gdyby go ręcznie tu wyliczyc.
            scores['train_accuracy'].append(logs['accuracy'])
            
            if verbose > 1:
                print(f"Train time: {scores['train_time'][-1]}, Epoch time: {end_epoch - self.start_epoch}")
                print(f"Train loss: {scores['train_loss'][-1]}, Test loss: {scores['loss'][-1]}")
                print(f"Train accu: {scores['train_accuracy'][-1]}, Test accu: {scores['accuracy'][-1]}")
    model.fit(
        x_train, y_train, 
        batch_size=batch_size, epochs=max_epochs,
        verbose=0,
        callbacks=[PredictionCallback()]
    )
    return scores




