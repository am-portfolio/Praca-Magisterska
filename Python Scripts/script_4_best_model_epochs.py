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
PLOTS_OUT = 'out_main/extra_plots'
os.makedirs(PLOTS_OUT, exist_ok=True)

# Dane dla wszystkich epok
df = loadDataFrame('out_main', 'best_by_classifier.summary')
df = df.rename(columns={'accuracy': 'acc'})
df = df[df.classifier_name == 'SalamonCNN'].reset_index(drop=True)
model_name = df.loc[0,:]['model_name']
classifier_name = df.loc[0,:]['classifier_name']
series = df.loc[0,:]['series']

# Dane najlepszego modelu dla wszystkich epok i wszystkich foldów
df_all = collectDataframe(os.path.join('out', classifier_name, f'SERIES {series}', model_name))
df_all = df_all.rename(columns={'accuracy': 'acc', 'train_accuracy': 'train_acc'})

# Dane najlepszego modelu dla wszystkich epok i urednionych foldów
df = loadDataFrame('out_main', 'all.summary')
df = df[(df.model_name == model_name) & (df.series == series)]
df = df.rename(columns={'accuracy': 'acc', 'train_accuracy': 'train_acc'})





# --------------------- EPOKI DLA RÓŻNYCH FOLDÓW ----------------------- #

def epochsPlot(score = 'acc', score_name = 'ACC'):
    MIN_EPOCH=1
    MAX_EPOCH=30
    
    plt.figure(figsize=(8.1,2.5))
    
    # EPOKI DLA RÓŻNYCH FOLDÓW
    plt.subplot(1,2,1)
    
    max_epochs = []
    min_accs = []
    max_accs = []
    for fold, df2 in df_all.groupby('fold'):
        df2 = df2.sort_values('epoch', ascending=False)
        df2 = df2[df2.epoch <= MAX_EPOCH]
        plt.plot(df2["epoch"].values, df2[score].values, color=getTabColor(fold), label=fold+1)  
        max_epochs.append(np.max(df2["epoch"].values))
        min_accs.append(df2[df2.epoch >= MIN_EPOCH][score].min())
        max_accs.append(df2[score].max())
    plt.xlim([MIN_EPOCH, np.max(max_epochs)])
    plt.ylim([np.min(min_accs), np.max(max_accs)])
    plt.xlabel('Epoka')
    plt.ylabel(f"$ {score_name} $")
    plt.legend(fontsize='small', ncol=5, loc='lower right')
    plt.title('Wszystkie wyniki walidacji krzyżowej')
        
    # UŚREDNONE EPOKI
    plt.subplot(1,2,2)
    df2 = df.sort_values('epoch', ascending=False)
    plt.plot(df2["epoch"].values, df2[score].values)  
    plt.xlim([MIN_EPOCH, np.max(max_epochs)])
    plt.ylim([np.min(min_accs), np.max(max_accs)])
    plt.xlabel('Epoka')
    plt.ylabel(f"$ \overline{{{score_name}}} $")
    plt.title('Średnia z 6 środkowych wyników')
    
    plt.tight_layout()

epochsPlot('acc', 'ACC')
plt.savefig(os.path.join(PLOTS_OUT, f'AllFoldsVsAvgEpoch.png'))
plt.close()





# --------------------- WYNIKI Z NAJLEPSZEGO MODELU ----------------------- #


df2 = df.sort_values('epoch', ascending=False)

plt.figure(figsize=(8,4))

# DOKŁADNOSCI
for i, (MIN_EPOCH, MAX_EPOCH) in enumerate([(0,10), (10, 150)]):
    df3 = df2[df2.epoch <= MAX_EPOCH]
    df3 = df3[df2.epoch >= MIN_EPOCH]
    plt.subplot(2,2,i+1)
    for score, score_name in [('acc', 'Zbiór testowy'), ('train_acc', 'Zbiór uczący')]:
        plt.plot(df3["epoch"].values, df3[score].values, label=score_name)  
    plt.legend(fontsize='small', ncol=5, loc='lower right')
    plt.xlim([MIN_EPOCH, MAX_EPOCH])
    plt.ylabel(f"$ \overline{{ACC}} $")
    plt.xlabel('Epoka')
    plt.title('Dokładnosć klasyfikacji')

# STRATA
for i, (MIN_EPOCH, MAX_EPOCH) in enumerate([(0,10), (10, 150)]):
    df3 = df2[df2.epoch <= MAX_EPOCH]
    df3 = df3[df2.epoch >= MIN_EPOCH]
    plt.subplot(2,2,i+3)
    for score, score_name in [('loss', 'Zbiór testowy'), ('train_loss', 'Zbiór uczący')]:
        plt.plot(df3["epoch"].values, df3[score].values, label=score_name)  
    plt.legend(fontsize='small', ncol=5, loc='upper right')
    plt.xlim([MIN_EPOCH, MAX_EPOCH])
    plt.ylabel(f"$ \overline{{LOSS}} $")
    plt.xlabel('Epoka')
    plt.title('Strata')
    
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUT, f'TestTrainAccOverEpochsBestBestCNN.png'))
plt.close()
