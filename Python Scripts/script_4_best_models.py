# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *


PLOTS_OUT = 'out_main/extra_plots'
os.makedirs(PLOTS_OUT, exist_ok=True)


# Dane dla wszystkich epok
df_all = loadDataFrame('out_main', 'all.summary')
df_all = df_all.sort_values('acc', ascending=False).reset_index(drop=True)

# Dane tylko dla najlepszych epok
df = df_all.drop_duplicates(["model_name", "series"]).reset_index(drop=True)
    
dfKNN = df[df.classifier_name == 'kNN'].head(150)
dfNN = df[df.classifier_name != 'kNN']

x = ['150 najlepszych klasyfikatorów kNN', 'Wszystkie przebadane sieci neuronowe']
y = [dfKNN['acc'].values, dfNN['acc'].values]

plt.figure(figsize=(8.5,5))
plt.subplot(2,1,1)
plt.boxplot(y, labels=x)
#plt.xlabel('Typ klasyfikatora')
plt.ylabel(f"$ \overline{{ACC}} $")
plt.title('Uśredniona dokładność przebadanych klasyfikatorów')
plt.axhline(y=np.max(y[0]),color='gray',linestyle='--', alpha=0.5)


# ----------- PUDEŁKA DLA NAJLEPSZYCH MODELI DANEGO TYPU ------------ #
      
# Dane dla wszystkich foldów
df = loadDataFrame('out_main', 'best_by_classifier.df')
df = df.rename(columns={'accuracy': 'acc'})

df2 = df.groupby('classifier_name')['acc'].apply(list).reset_index()
df2.loc[df2.classifier_name == 'SalamonCNN', 'classifier_name'] = 'SalamonCqtCNN'

x = df2['classifier_name'].values
y = df2['acc'].values

sort_order = np.array([np.median(v) for v in y]).argsort()
x = x[sort_order]
y = y[sort_order]

plt.subplot(2,1,2)
plt.boxplot(y, labels=x)
#plt.xlabel('Klasyfikator')
plt.ylabel(f"$ ACC $")
plt.title('Dokładność klasyfikacji zbiorów testowych dla najlepszych klasyfikatorów danego typu')
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.savefig(os.path.join(PLOTS_OUT, f'BestModelsByClassifier.png'))

plt.close()




# ----------- PUDEŁKA DLA NAJLEPSZYCH MODELI DANEGO TYPU ------------ #

# Dane dla wszystkich foldów
df = loadDataFrame('out_main', 'best_by_input.df')
df = df.rename(columns={'accuracy': 'acc'})
df = df[~df.input_name.isin(['mfccmax', 'mfccl2'])]

df2 = df.groupby('input_name')['acc'].apply(list).reset_index()
x = np.array([v.upper() for v in df2['input_name'].values])
y = df2['acc'].values

sort_order = np.array([np.median(v) for v in y]).argsort()
x = x[sort_order]
y = y[sort_order]

plt.figure(figsize=(8,2.5))
plt.boxplot(y, labels=x)
plt.xlabel('Własnosci wejściowe')
plt.ylabel(f"$ ACC $")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUT, f'BestModelsByInput.png'))
plt.close()