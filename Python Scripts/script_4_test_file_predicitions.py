# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *




PLOTS_OUT = 'out_main/extra_plots'
os.makedirs(PLOTS_OUT, exist_ok=True)





# Dane
df = loadDataFrame('out_main', 'all_networks.summary')
# Etykiety
labels = loadDictionary('src', 'y_test_data.npy')['labels']

# Upewnienie się że dla wszystkich dnaych jest to samo
count = 0
for fn in df["file_names"]:
    count += (fn == df["file_names"][0]).all()
assert count == len(df)

# Zebranie informacji
series = df.iloc[0]
y_true = series['y_true']
file_names = series['file_names']
file_augs  = series['file_augs']

# Sprawdzenie predykcji
results = []
for index, series in df.iterrows():
    y_prob = series['y_prob']
    y_pred = np.argmax(y_prob, axis=1)
    results.append(y_true == y_pred)
results = np.stack(results, axis=1)
results = np.sum(results, axis=1)


# Sortowanie i podział
df = pd.DataFrame({
    'file_names': file_names,
    'file_augs': file_augs,
    'y_true': y_true,
    'results': results
})
df = df.sort_values(['y_true', 'file_names', 'file_augs'], ascending=[True, False, True])
dfs = [x for _, x in df.groupby('y_true')]
results_splits = [df['results'].values.reshape([1450//50, 50]) for df in dfs]


# Plotowanie
plt.figure(figsize=(8,4.5))

for i, label in enumerate(labels):
    if i > 7:
        plt.subplot(3,4, i+2)
    else:
        plt.subplot(3,4, i+1)
    sn.heatmap(results_splits[i], square=True, cbar=False, cmap="binary_r")
    plt.title(label)
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.tick_top()
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_OUT, f'FilePredictionsAccuracyInTop50Models.png'))



plt.figure(figsize=(6,6))
unique, counts = np.unique(df['results'], return_counts=True)
counts = counts / 14500
sums = np.cumsum(counts)
plt.plot(np.linspace(0, 1, len(sums)), sums)
plt.hlines(0.04, 0, 1)