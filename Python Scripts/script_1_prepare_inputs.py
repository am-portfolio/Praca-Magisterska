# SKRYPT GENERUJĄCY RÓŻNE WYKRESY UŻYTE W PRACY

import config
from pymodules.audiohelpers import *
from pymodules.utilities import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from timeit import default_timer as timer
import time
from tqdm import tqdm
import ntpath
import random




# Miejsce plików
src_path = 'src'
# Generator losowy ze stałym seedem.
randomizer = random.Random(13)



with open("out_main/Audio files details.txt", "w") as text_file:
    def FILE_PRINT(*args, **kwargs):
        print(*args, file=text_file, **kwargs)
        
        
    FILE_PRINT(f'Output from: script_1_prepare_inputs.py\n\n')  
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    
    
    # Przygotowanie listy klas plików (foldery w src)
    print('Getting labels...')
    labels = []
    for path in os.listdir(src_path):
        if os.path.isdir(os.path.join(src_path, path)):
            labels.append(path)
    labels = np.array(labels)
    
    
    
    
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    # Przygotowanie listy plików dla każdej klasy (w losowej kolejnosci wg seedu).
    print('Getting paths for each label...')
    files = []
    for label in labels:
        found_files = librosa.util.find_files(os.path.join(src_path, label))
        randomizer.shuffle(found_files)
        files.append(found_files)
    
    # Informacje o ilosci plików na klasę
    plt.figure(figsize=(6,4))
    files_per_label = np.array([len(f) for f in files])
    max_files = np.max(files_per_label)
    sorting = np.argsort(files_per_label)
    plt.barh(labels[sorting], files_per_label[sorting])
    plt.xlim(0, max_files)
    for i, v in enumerate(files_per_label[sorting]):
        plt.gca().text(20, i-0.13, f'{v} ({np.round(v/np.sum(files_per_label)*100,2)}%)', color='white', fontweight='bold', fontsize="8.5")
    plt.tight_layout()
    plt.savefig('plots/files_amount.png')
    plt.close()
    
    print('Getting files stats...')
    # Inne statystyki
    files_channels = []
    files_duration = []
    files_samplerate = []
    files_subtype = []
    for ff in tqdm(files):
        for f in ff:
            data = sf.info(f)
            files_channels.append(data.channels)
            files_duration.append(data.duration)
            files_samplerate.append(data.samplerate)
            files_subtype.append(data.subtype)
    files_channels = np.array(files_channels)
    files_duration = np.array(files_duration)
    files_samplerate = np.array(files_samplerate)
    files_subtype = np.array(files_subtype)
    files_total = int(np.sum(files_per_label))
    
    FILE_PRINT('\n\n--- TOTAL FILES ---')
    FILE_PRINT(files_total)
    
    def printStats(data, name):
        FILE_PRINT(f'\n\n--- {name} ---')
        unique, counts = np.unique(data, return_counts=True)
        for i in range(len(unique)):
            FILE_PRINT(f'{unique[i]}:\t{counts[i]} ({np.round(counts[i]/np.sum(counts)*100,2)}%)')
    printStats(files_channels, 'CHANNELS')
    printStats(files_samplerate, 'SAMPLE RATES')
    printStats(files_subtype, 'TYPE')
    
    FILE_PRINT('\n\n--- DURATION ---')
    FILE_PRINT(f'mean:\t{np.round(np.mean(files_duration),2)}')
    FILE_PRINT(f'median:\t{np.round(np.median(files_duration),2)}')
    FILE_PRINT(f'std:\t{np.round(np.std(files_duration),2)}')
    FILE_PRINT(f'min:\t{np.round(np.min(files_duration),2)}')
    FILE_PRINT(f'max:\t{np.round(np.max(files_duration),2)}')
    
    
    
    
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    # INFORMACJE O AUGMENTACI
    number_of_folds = 10
    files_per_split = (files_per_label/number_of_folds).astype(int)
    augms_per_split = -(files_per_split - np.max(files_per_split)).astype(int)
    augms_per_file  = np.ceil(augms_per_split/files_per_split).astype(int)
    
    FILE_PRINT('\n\n--- FILES PER SPLIT ---')
    for l, label in enumerate(labels):
        FILE_PRINT(f'{label}:\t{files_per_split[l]}')
    FILE_PRINT('\n\n--- AUGMENTATIONS PER SPLIT ---')
    for l, label in enumerate(labels):
        FILE_PRINT(f'{label}:\t{augms_per_split[l]}')
    FILE_PRINT('\n\n--- MAX. AUGMENTATIONS PER FILE ---')
    for l, label in enumerate(labels):
        FILE_PRINT(f'{label}:\t{augms_per_file[l]}')
    
    
    
    
    
    
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    # Indeksy do foldów
    data_count = max_files*len(labels)
    data_per_fold = int(data_count / number_of_folds)
    fold_indices = np.zeros([number_of_folds, data_count]).astype(bool)
    for fold in range(number_of_folds):
        beg = fold * data_per_fold
        end = (fold+1) * data_per_fold
        fold_indices[fold, beg:end]=True
    fold_indices = ~fold_indices
    
    
    
    
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    load_times = []
    
    def getAugmArray(label, randomizer):
        a = np.array([1,-1,2,-2])
        randomizer.shuffle(a)
        if(label=='HHatO' or label=='HHatC'):
            a = a/2
        return a
    
    def createTrainData(saveas, audioToInput, randomizer):
        files_done = 0
        
        # Przygotowanie danych wejciowych i ich augmentacji
        # - cała tablica zawierająca wszystkie foldy i augmentacje
        # - augmentacjenie przeciekają na inne foldy
        # - zbalansowany zbiór uczący po 1450 przykładów na klasę i po 1450 na fold
        
        data = []
        data_augm = []
        data_names = []
        data_labels = []
        data_times = []
        
        z = np.zeros([10,10])
        for fold in range(number_of_folds):
            for l, label in enumerate(labels):
                # Kolejny zakes plików labelu
                beg = fold*files_per_split[l]
                end = (fold+1)*files_per_split[l]
                paths = files[l][beg:end]
                
                # Generowanie augmentacji i inputów
                augms_done = 0
                for path in paths:
                    # Wczytanie pliku
                    start = timer()
                    audio, _ , _ = loadSample(path)
                    end = timer()
                    load_times.append(end-start)
                    # Przetwarzanie danych
                    start = timer()
                    S = audioToInput(audio)
                    end = timer()
                    # Zapisanie danych
                    data.append(S)
                    data_augm.append(0)
                    data_times.append(end-start)
                    data_names.append(ntpath.basename(path))
                    data_labels.append(l)
                    
                    # Augmentacje
                    augms_settings = getAugmArray(label, randomizer)
                    for augm in range(augms_per_file[l]):
                        if(augms_done >= augms_per_split[l]):
                            break;
                        # Augmentowanie danych
                        aug_amount = augms_settings[augm]
                        aug_audio = interpSample(audio, aug_amount)
                        # Przetwarzanie danych
                        start = timer()
                        S = audioToInput(aug_audio)
                        end = timer()
                        # Zapisanie danych
                        data.append(S)
                        data_augm.append(aug_amount)
                        data_times.append(end-start)
                        data_names.append(ntpath.basename(path))
                        data_labels.append(l)
                        
                        augms_done = augms_done + 1
                        
                    files_done = files_done + 1
                    print(f'[{saveas}] FILES DONE {files_done}/{files_total} ({files_done/files_total*100}%)')
        
        # Walidacja
        assert len(data) == data_count
        
        print('\n\nStacking data...')
        data = np.stack(data, axis=0).astype(np.float32)
        data_augm = np.array(data_augm)
        data_names = np.array(data_names)
        data_times = np.array(data_times)
        data_labels = np.array(data_labels)
        
        # ZAPISANIE DANYCH
        print('\n\Saving input data...')
        savedata = {
            "x":            data,
            "y":            data_labels,
            "labels":       labels,
            "names":        data_names,
            "augmented":    data_augm,
            "folds":        fold_indices,
            "times":        data_times
        }
        np.save(os.path.join(src_path, saveas), savedata)
        return
        
    
    
    #---------------------------------------------------#
    #---------------------------------------------------#
    
    if(False):
        # Przygotowanie danych
        # Za każdym razem seed jest resetowany by każda metoda
        # miała te same augmentacje
        print('\n\nPreparing train data...')
        
        randomizer = random.Random(13)
        createTrainData('cqcc_train_data.npy', audioToCQCC, randomizer)
        
        randomizer = random.Random(13)
        createTrainData('mfcc_train_data.npy', audioToMFCC, randomizer)
        
        randomizer = random.Random(13)
        createTrainData('mfccl2_train_data.npy', audioToMFCCL2, randomizer)
        
        randomizer = random.Random(13)
        createTrainData('mfccmax_train_data.npy', audioToMFCCMAX, randomizer)
        
        randomizer = random.Random(13)
        createTrainData('mels_train_data.npy', audioToMELS, randomizer)
        
        randomizer = random.Random(13)
        createTrainData('cqt_train_data.npy', audioToCQT, randomizer)
        
        # Czasy wczytywania i interpolacji
        load_times = np.array(load_times)
        np.save(os.path.join(src_path, 'load_times.npy'), load_times)
        
        
        
        
        
        #---------------------------------------------------#
        #---------------------------------------------------#
        
        # Upewnienie się że wszystko wylosowano tak samo.
        print('Validating outputs...')
        data1 = loadDictionary(src_path, 'mfcc_train_data.npy')
        for x_name in ['mfccl2', 'mfccmax', 'mels', 'cqt', 'cqcc']:
            data2 = loadDictionary(src_path, x_name + '_train_data.npy')
            assert (data1['y']==data2['y']).all() 
            assert (data1['names']==data2['names']).all() 
            assert (data1['augmented']==data2['augmented']).all() 
        data = data1
        
        
        #---------------------------------------------------#
        #---------------------------------------------------#
        
        # Przygotowanie pliku z wynikami predykcji zbioru walidującego
        # dla różnych foldów oraz innmi metadanymi.
        savedata = {
            "labels":       labels,
            
            "y":            [], # [n,m] poprawna etykieta m-tego przykładu w n-tym foldzie 
            "names":        [], # Jak wyżej ale nazwa pliku
            "augmented":    [], # Jak wyżej ale siła augmentacji
        }
        for fold in data['folds']:
            savedata['y'].append(data['y'][~fold])
            savedata['names'].append(data['names'][~fold])
            savedata['augmented'].append(data['augmented'][~fold])
        
        saveObject(src_path, 'y_test_data.npy', savedata)