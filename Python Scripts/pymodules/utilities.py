import numpy as np
import multiprocessing
from tqdm import tqdm
from termcolor import colored
import os
import pandas as pd

# Zarządzanie paletą kolorów matplotlib
def getTabColor(i = 'next'):
    if(i == 'next'):
        getTabColor.i += 1
        i = getTabColor.i
    elif(i == 'previous'):
        i = getTabColor.i
    return getTabColor.list[i % len(getTabColor.list)]
getTabColor.list = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]
getTabColor.i = -1

# Printuje kolorowy tekst w konsoli (w formie nagłówka)
def printHeader(text):
    color = 'cyan'
    padding = "=" * 6
    print()
    print(colored("=" * (len(text) + len(padding)*2 + 2), color))
    print(colored(f"{padding} {text} {padding}", color))
    print(colored("=" * (len(text) + 12 + 2), color))
    print()
def printWarning(text):
    color = 'magenta'
    print(colored(text, color))
    
def printInfo(text):
    color = 'cyan'
    print(colored(text, color))
    
# Sprowadza wartości z data do zakresu [min, max]
def normalize(data, min=0, max=1):
    assert min < max
    data_min = np.min(data)
    data_max = np.max(data)
    old_range = data_max - data_min
    new_range = max - min
    return (data - data_min)*new_range/old_range + min

# Wykonuje mapę w wielu wątkach
def threadedMap(fun, data):
    pool = multiprocessing.Pool()
    results = pool.map(fun, data)
    return results

# Zapisuje tekst do pliku
def saveText(out_path, filename, text):
    with open(os.path.join(out_path, filename), "w") as text_file:
        print(text, file=text_file)

# Zapisuje obiekt do pliku
def saveObject(out_path, filename, data):
    np.save(os.path.join(out_path, filename), data)
    
# Wczytuje discrionary zapisane saveObject()
def loadDictionary(src_path, filename):
    return np.load(os.path.join(src_path, filename), allow_pickle=True).item()

# Zapisuje dataframe do pliku
def saveDataFrame(out_path, filename, df):
    os.makedirs(out_path, exist_ok=True)
    df.to_pickle(os.path.join(out_path, filename))
    
# Wczytuje dataframe z pliku
def loadDataFrame(src_path, filename):
    return pd.read_pickle(os.path.join(src_path, filename))