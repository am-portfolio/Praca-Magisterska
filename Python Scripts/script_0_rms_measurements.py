"""
    ANALIZA DANYCH POD KĄTEM WYZNACZENIA WYSTARCZALNEJ DŁUGOŚCI
    DLA BADANIA DŹWIĘKÓW
"""

import config
from pymodules.audiohelpers import *
from pymodules.utilities import *
import multiprocessing
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import sounddevice as sd
import time
from tqdm import tqdm



# Przykłądowy wykres RMS:
calcuateRealLength('src/Kick/Kick 0000.flac', plot=True)
plt.savefig('plots/RMS example.png')
   



tresholds = [0.25, 0.1, 0.05, 0.01]
files = librosa.util.find_files('src')
    
print('CALCULATING TIMES...')
output = []
for i, file in enumerate(files):
    print(f'{i+1}/{len(files)}')
    output.append(calcuateRealLength(file, tresholds))
output = np.vstack(output)

# Wykres czasów:
with open("out_main/RMS tresholds.txt", "w") as text_file:
    print(f'Output from: script_0_rms_measurements.py\n\n', file=text_file)
    
    plt.figure(figsize=(7, 3))
    percentage = 0.95
    plt.hlines(percentage, 0, len(files)+1, color='black', alpha=0.3, label=f'{percentage*100}%')
    
    
    for i in range(len(tresholds)):
        treshold = tresholds[i]
        times = np.sort(output[:,i])
        print('TRESHOLD: ', treshold, file=text_file)
        print('Min: ', np.round(np.min(times), 3), ', Max: ', np.round(np.max(times), 3), ', Avg: ', np.round(np.mean(times), 3), file=text_file)
        times = np.insert(times, 0, 0)
        count = np.linspace(0, 1, len(times))
        plt.step(times, count, where='post', color=getTabColor(i+1))
        plt.vlines(times[int(len(times)*percentage)], 0, 1, color=getTabColor(i+1),
                   alpha=0.8, linestyle='--', label=f'Próg: {int(treshold*100)}% z RMS max')
    
    plt.xlim(0, 3)
    plt.ylabel('Procent dźwięków krótszych')
    plt.xlabel('Czas [s]')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/RMS tresholds.png')