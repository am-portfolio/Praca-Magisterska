# Aron Mandrella – Praca Magisterska (2020)
## Opis
Celem pracy było znalezienie sztucznej sieci neuronowej służącej do klasyfikacji dźwięków perkusyjnych zapewniającej wysoką dokładność. Największą uwagę przyłożono do konwolucyjnych sieci neuronowych. W ramach badań testowano różne metody reprezentacji dźwięku i różne techniki trenowania sieci neuronowych.
* [Pełen tekst pracy magisterskiej](https://github.com/am-portfolio/Praca-Magisterska/blob/main/AMandrella%20-%20Praca%20Magisterska.pdf)
* [Wyniki (na dysku Google)](https://drive.google.com/drive/folders/1CWwUyckJevgqcemdiRQTdpQhYnwwuz_g?usp=sharing)
## 🧰 Wykorzystane technologie i narzędzia
* **Python**
* **TensorFlow 2, Librosa, Matplotlib, NumPy, Pandas, sklearn**
* **Spyder IDE**
## 🎓 Zdobyta bądź poszerzona wiedza
* Umiejętność pozyskiwania informacji z angielskiej literatury naukowej
* Teoria z zakresu uczenia maszynowego i sieci neuronowych
* Metody normalizacji danych
* Algorytmy gradientowe
* Nowe metody uczenia sieci neuronowych (dropout, batch normalisation) 
* Metody analizy i reprezentacji dźwięku (transformacja Fouriera, spektrogram, transformacja ze stałym Q, transformata cosinusowa, współczynniki mel-cepstralne)
* Metody walidacji modeli stworzonych metodą uczenia maszynowego (walidacja krzyżowa, tablice pomyłek, wskaźniki jakości klasyfikacji itp.)
* Akademickie metody statystycznej analizy zebranych danych (np. wykresy pudełkowe)
```python
# Moduły napisane na potrzeby pracy
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
```
