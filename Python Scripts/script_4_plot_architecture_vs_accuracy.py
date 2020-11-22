# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *



# Ścieżka wyjsciowa
PLOTS_OUT = 'out_main/hyperparameters_vs_acc'
os.makedirs(PLOTS_OUT, exist_ok=True)
      



# --------------------- PRZYGOTOWANIE DANYCH ----------------------- #

# Dane dla wszystkich epok
df_all = loadDataFrame('out_main', 'all.summary')
df_all = df_all.sort_values('acc', ascending=False).reset_index(drop=True)

# Dane tylko dla najlepszych epok
df = df_all.drop_duplicates(["model_name", "series"]).reset_index(drop=True)
    



# --------------------- ANALIZA ALGORYTMU kNN ----------------------- #

def kNN(df, LIMIT = 5):   
    # Filtracja na kNN
    df = df[df.classifier_name == 'kNN']
    
    
    # Ploty
    plt.figure(figsize=(8,4))
    
    plt.subplot(2,2,1)
    plotBy(df, "input_name", "Własności uczące", limit=LIMIT)
    plt.xticks(rotation=25)
    
    plt.subplot(2,2,2)
    plotBy(df, "k", "Liczba sąsiadów (k)", astype='int', limit=LIMIT)
    
    plt.subplot(2,2,3)
    plotBy(df[df.input_name == "cqcc"], "row_limit", "Limit wierszy gdy użyto CQCC", astype='int', limit=LIMIT)
    
    plt.subplot(2,2,4)
    plotBy(df[df.input_name == "mfcc"], "row_limit", "Limit wierszy gdy użyto MFCC", astype='int', limit=LIMIT)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'kNN_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
kNN(df_all, LIMIT=4)





# --------------------- ANALIZA ALGORYTMU DNN ----------------------- #

def DNN(df, LIMIT = 4, MAX_EPOCHS = None):    
    # Filtracja na DNN
    df = df[df.classifier_name == 'DNN']
    
        
    # Dane dla 1 serii pomiarów
    df2 = df[df.series == 1]
    if MAX_EPOCHS != None:
        df2 = df2[df2.epoch <= MAX_EPOCHS]
    # Ploty dla 1 serii pomiarów
    plt.figure(figsize=(8,4))
    
    plt.subplot(2,2,1)
    plotBy(df2, "input_name", "Własności uczące", limit=LIMIT)
    
    plt.subplot(2,2,2)
    plotBy(df2, "hidden_layers", "Ukryte warstwy", limit=LIMIT, astype='int')
    
    plt.subplot(2,2,3)
    plotBy(df2, "layer_neurons", "Neurony w warstwie", limit=LIMIT, astype='int')
    
    plt.subplot(2,2,4)
    plotEpchs(df2)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'DNN_S1_HpVsAcc - LIMIT {LIMIT}, MAX EPOCHS {MAX_EPOCHS}.png'))
    plt.close()
    
    
    # Dane dla 2 serii pomiarów
    df2 = df[(df.hidden_layers == 2) & (df.layer_neurons >= 256) & (df.series <= 2)]
    if MAX_EPOCHS != None:
        df2 = df2[df2.epoch <= MAX_EPOCHS]
    # Ploty dla 2 serii pomiarów
    plt.figure(figsize=(8,4))
    
    plt.subplot(2,2,1)
    plotBy(df2, "input_name", "Własności uczące", limit=LIMIT)
    
    plt.subplot(2,2,2)
    plotBy(df2, "layer_neurons", "Neurony w warstwie", limit=LIMIT, astype='int')
    
    plt.subplot(2,2,3)
    plotBy(df2[df2.input_name == "cqcc"], "row_limit", "Limit wierszy gdy użyto CQCC", astype='int', limit=LIMIT)
    
    plt.subplot(2,2,4)
    plotBy(df2[df2.input_name == "mfcc"], "row_limit", "Limit wierszy gdy użyto MFCC", astype='int', limit=LIMIT)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'DNN_S2_HpVsAcc - LIMIT {LIMIT}, MAX EPOCHS {MAX_EPOCHS}.png'))
    plt.close()
    
    
    
    # Dane dla 3 serii pomiarów
    df2a = df[(df.series == 2) & (df.layer_neurons >= 512) & (df.row_limit.isin([20,40]))]
    if MAX_EPOCHS != None:
        df2a = df2a[df2a.epoch <= MAX_EPOCHS]
    df2b = df[df.series == 3]
    if MAX_EPOCHS != None:
        df2b = df2b[df2b.epoch <= MAX_EPOCHS]
    df2 = pd.concat([df2a, df2b])
    # Ploty dla 3 serii pomiarów
    plt.figure(figsize=(5,2))

    # plt.subplot(2,2,3)
    # df2.loc[:, "input_name"] = df2["input_name"] + '-' + df2["row_limit"].apply(lambda a: str(int(a)))
    # plotBy(df2, "input_name", "Własności uczące i liczba wierszy", limit=LIMIT)
    
    plt.subplot(1,2,1)
    plotBy(df2, "l2", "Regularyzacja L2", limit=LIMIT)
    
    plt.subplot(1,2,2)
    plotBy(df2, "dropout", "Dropout", limit=LIMIT)
    
    # plt.subplot(2,2,4)
    # plotEpchs(df2)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'DNN_S3_HpVsAcc - LIMIT {LIMIT}, MAX EPOCHS {MAX_EPOCHS}.png'))
    plt.close()
    
DNN(df_all, LIMIT = 4, MAX_EPOCHS = 50)






# ------------ ANALIZA ALGORYTMÓW HuzaifahCNN, GajhedeCNN i SalamonCNN ----------------- #

def PaperCNN(df, classifier_name, optimizer, LIMIT = 5, MAX_EPOCHS = None, MIN_EPOCH = 10):    
    # Filtracja na DNN
    df = df[(df.classifier_name == classifier_name) & (df.optimizer == optimizer)]
    

    # Dane dla 1 serii pomiarów
    df2 = df
    def rowLimitToStr(row_limit):
        if row_limit == None:
            return ''
        elif np.isnan(row_limit):
            return ''
        else:
            return f'-{int(row_limit)}'
    df2.loc[:, "input_name"] = df2["input_name"] + df2["row_limit"].apply(rowLimitToStr)

    
    # Ploty dla 1 serii pomiarów
    plt.figure(figsize=(5,3))
    
    plt.subplot(2,1,1)
    plotBy(df2, "input_name", "Własności uczące", limit=LIMIT, kind='scatter')
    
    plt.subplot(2,1,2)
    plotEpchs(df2, True, MIN_EPOCH)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'{classifier_name}_{optimizer}_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
    
PaperCNN(df_all, 'HuzaifahCNN', 'adam', LIMIT = None)
PaperCNN(df_all, 'GajhedeCNN', 'adam', LIMIT = None)
PaperCNN(df_all, 'SalamonCNN', 'sgd', LIMIT = None, MIN_EPOCH=100)
PaperCNN(df_all, 'SalamonCNN', 'adam', LIMIT = None)




# ----------------- DALSZA ANALIZA ALGORYTMU SalamonCNN ----------------- #

def BestCNN(df):    
    # Filtracja na DNN
    df = df[df.classifier_name == 'BestCNN']
    
        
    # Dane dla 1 serii pomiarów
    LIMIT = None
    df2 = df[df.series == 1]
    # Ploty dla 1 serii pomiarów
    # plt.figure(figsize=(8,4))
    plt.figure(figsize=(8,2.5))

    plt.subplot(1,3,1)
    plotBy(df2, "batchnorm", "Normalizacja mini-serii", limit=LIMIT, astype='bool', kind='scatter')
    
    plt.subplot(1,3,2)
    plotBy(df2, "l2", "Regularyzacja L2", limit=LIMIT, kind='scatter')
    
    plt.subplot(1,3,3)
    plotBy(df2, "dropout", "Dropout", limit=LIMIT, kind='scatter')
    
    # plt.subplot(2,2,4)
    # plotEpchs(df2, True)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'BestCNN_S1_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
    
    
    # Dane dla 2 serii pomiarów
    LIMIT = None
    df2 = df[df.series <= 3]
    df2 = df2[(df.filters == (24, 48, 48)) & (df.l2 == 0.001) & (df.dropout == True) & (df.batchnorm == False)]
    df2["pool_size"] = df2["pool_size"].apply(str)
    # Ploty dla 2 serii pomiarów
    plt.figure(figsize=(3.5,3))
    
    plt.subplot(2,1,1)
    plotBy(df2, "pool_size", "Rozmiar maski poolingowej", limit=LIMIT, kind='scatter')
    
    plt.subplot(2,1,2)
    plotEpchs(df2, legend=True)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'BestCNN_S2_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
    
        
    # Dane dla 3 serii pomiarów
    LIMIT = None
    df2 = df[df.series <= 3]
    df2 = df2[df.series.isin([1,3]) & (df.l2 == 0.001) & (df.dropout == False) & (df.batchnorm == True)]
    df2["filters"] = df2["filters"].apply(str)
    # Ploty dla 2 serii pomiarów
    plt.figure(figsize=(5,3))
    
    plt.subplot(2,1,1)
    plotBy(df2, "layer_neurons", "Liczba neuronów", limit=LIMIT, kind='scatter', astype='int')
    
    plt.subplot(2,1,2)
    plotBy(df2, "filters", "Rozmiary filtrów", limit=LIMIT, kind='scatter')
    
    # plt.subplot(1,3,3)
    # plotEpchs(df2)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'BestCNN_S3_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
    
    # Dane dla 4 serii pomiarów
    LIMIT = None
    df2 = df[df.series == 4]
    df2["filters"] = df2["filters"].apply(str)
    df2["pool_size"] = df2["pool_size"].apply(str)
    # Ploty dla 4 serii pomiarów
    plt.figure(figsize=(8,4))
    
    plt.subplot(2,2,1)
    plotBy(df2, "layer_neurons", "Liczba neuronów", limit=LIMIT, kind='scatter', astype='int')
    
    plt.subplot(2,2,2)
    plotBy(df2, "pool_size", "Rozmiar maski poolingowej", limit=LIMIT, kind='scatter')
    
    plt.subplot(2,2,3)
    df2['batchnorm'] = df2['batchnorm'].apply(lambda v: 'Normalizacja\nmini-serii' if v else 'Dropout')
    plotBy(df2, "batchnorm", "Zapobieganie nadmiernemu dopasowaniu", limit=LIMIT, kind='scatter')

    plt.subplot(2,2,4)
    plotEpchs(df2, True)
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUT, f'BestCNN_S4_HpVsAcc - LIMIT {LIMIT}.png'))
    plt.close()
    
    
    # Dodatkowe
    
    
BestCNN(df_all)





# ----------------- WYPŁYW LICZBY PARAMETRÓW NA WYNKIKI ----------------- #

df2 = df[df.classifier_name != 'kNN'].sort_values('trainable_params', ascending=False)
df2 = df2.groupby('classifier_name')

plt.figure(figsize=(6,3))
for index, dfg in df2:
    x = dfg['trainable_params'].values
    y = dfg['acc'].values
    plt.scatter(x,y, marker='.', label=index)
    

plt.xlabel('Liczba trenowalnych parametrów')
plt.ylabel(f"$ \overline{{ACC}} $")
plt.tight_layout()
plt.legend(ncol=3, fontsize='small')
plt.savefig(os.path.join(PLOTS_OUT, f'TrainableParamsVsAcc.png'))
plt.close()