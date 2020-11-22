# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *


PLOTS_OUT = 'out_main'

y_test_data = loadDictionary('src', 'y_test_data.npy')

def ConfusionMatrixVs(series1, name1, series2, name2):
    # Labele, predykcje i tablice
    labels = y_test_data["labels"]
    y_true1 = series1["y_true"]
    y_prob1 = series1["y_prob"]
    y_pred1 = np.argmax(y_prob1, axis=1)
    cnf1 = sklearn.metrics.confusion_matrix(y_true1, y_pred1, normalize='true')
    y_true2 = series2["y_true"]
    y_prob2 = series2["y_prob"]
    y_pred2 = np.argmax(y_prob2, axis=1)
    cnf2 = sklearn.metrics.confusion_matrix(y_true2, y_pred2, normalize='true')
    
    save_to = os.path.join(PLOTS_OUT, 'extra_plots')
    
    # Przejcie z cnf2 -> cnf1
    cnfm = (cnf1 - cnf2) * 100
    fmt = '0.1f'
    vmax = np.max(np.abs(cnfm))
    vmin = -vmax
    # Wykres:
    plt.figure(figsize=(5,5))
    cnf_matrix_df = pd.DataFrame(cnfm, index=labels, columns=labels)
    sn.heatmap(cnf_matrix_df, vmax=vmax, vmin=vmin, annot=True, fmt=fmt, cbar=False, square=True, cmap="RdBu")
    plt.xlabel('Predykcja')
    plt.ylabel('Poprawna etykieta')
    plt.gca().xaxis.tick_top()
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.xticks(rotation=45) 
    plt.title(f'{name2} → {name1}')
    for _ in range(4):
        plt.tight_layout()
    os.makedirs(save_to, exist_ok=True)
    plt.savefig(os.path.join(save_to, f'{name2}_vs_{name1}_ConfusionMatrix.png'))




# ----------- PUDEŁKA DLA NAJLEPSZYCH MODELI DANEGO TYPU ------------ #
      
# Dane dla wszystkich foldów
df = loadDataFrame('out_main', 'all_networks.summary')
df = df.rename(columns={'accuracy': 'acc'})
df = df[df.classifier_name == 'BestCNN']
df = df[df.series <= 3]
df = df[(df.filters == (24, 48, 48)) & (df.l2 == 0.001) & (df.dropout == True) & (df.batchnorm == False)]

ConfusionMatrixVs(df[df.pool_size == (3,2)].iloc[0], '(3,2)', df[df.pool_size == (2,3)].iloc[0], '(2,3)')
plt.close()



# ----------- BATCHNORM I DROPOUT ------------ #
      
plt.figure(figsize=(5,2))
df = loadDataFrame('out_main', 'all.summary')
df = df.sort_values('acc', ascending=False).reset_index(drop=True)
df = df.drop_duplicates(["model_name", "series"]).reset_index(drop=True)
df = df[df.classifier_name == 'BestCNN']
df = df[df.series.isin([1,4])]
df['diff'] = df['acc'] - df['train_acc']
df = df.sort_values('diff', ascending=False)
def dpbpname(s):
    if s == 'DP+':
        return 'DP'
    if s == '+BN':
        return 'BN'
    if s == '+':
        return 'Brak'
    return s
df['gen'] = (df['dropout'].apply(lambda v: 'DP' if v else '') + '+'
             + df['batchnorm'].apply(lambda v: 'BN' if v else '')).apply(dpbpname)
plotBy(df, 'gen', 'Metody zapobiegania nadmiernemu dopasowaniu',
       kind='scatter', metric='diff', metric_name='ACC - TrainACC')
plt.tight_layout()
plt.savefig(os.path.join('out_main/extra_plots', f'OverfitGeneralization.png'))
plt.close()



# ----------- CZASY ------------ #

df = loadDataFrame('out_main', 'best_by_classifier.df')
df = df[df.classifier_name == 'SalamonCNN']
prediction_times = df['test_time']
t_train = df['train_time'].mean()
load_times = np.load('src/load_times.npy', allow_pickle=True)
cqt_times = loadDictionary('src', 'cqt_train_data.npy')['times']


t_pred = np.mean(prediction_times)*1000
t_load = np.mean(load_times)*1000
t_cqt  = np.mean(cqt_times)*1000

labels = ['Wczytywanie i wstępne przetwarzanie', 'Wyznaczenie CQT', 'Predykcja']
sizes = [t_load, t_cqt, t_pred]
explode = (0, 0, 0.1)

plt.figure(figsize=(5,2))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
