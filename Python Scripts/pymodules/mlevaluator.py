# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from timeit import default_timer as timer
from .utilities import *
from .mlrnhelpers import *
from sklearn.model_selection import ParameterGrid
import pandas as pd
from datetime import datetime
from pprint import pprint
import ntpath
import re


# Wyciąga z podsumowania modelu informację o trenowalnych parametrach
def summaryToTrainableParams(dir_path, file):
    if file.startswith('kNN'):
        return None
    # Wczytanie pliku
    summary_text = None
    f = open(os.path.join(dir_path, file), "r")
    summary_text = f.read() 
    f.close() 
    # Wyniki RegEx
    reg_out = re.findall("Trainable params: ([0-9,.]+)", summary_text)
    # Konwersja na liczbę
    return int(reg_out[0].replace(',', ''))


# Łączy pliki.npy we wskazanym folderze w jeden dataframe.
def collectDataframe(dir_path, load_predictions=True, stack_folds=False):
    
    EXPECTED_NMBER_OF_FOLDS = 10
    VECTOR_SCORES = ["loss", "train_loss", "accuracy", "train_accuracy", "train_time", "test_time"]
    MATRIX_SCORES = ["predictions"]
   
    ## -------------- ŁĄCZENIE WYNIKOW Z FOLDOW -------------- ##
    
    # Zbierze tu wyniki z wszystkich foldów dla wszystkich wariacji klasyfikatora
    models_scores = []
    
    # Wyniki foldów dla wszystkich wariacji klasyfikatora, nawet tych
    # które nie koniecznie były wykonane powyżej, a jakos wczesniej
    all_models_folds_files = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                if root not in all_models_folds_files:
                    all_models_folds_files[root] = []
                all_models_folds_files[root].append(file)
    
    
    
    # Łączenie foldów dla wszystkich modeli
    printInfo('LOADING AND STACKING SCORES')
    
    for folds_files_dir in all_models_folds_files.keys():
        print(folds_files_dir)
        
        folds_files = all_models_folds_files[folds_files_dir]
        # Numer serii
        series_number = int(ntpath.basename(os.path.dirname(folds_files_dir)).split()[-1])
        # Liczba trenowalnych parametrów
        trainable_params = summaryToTrainableParams(
            folds_files_dir, ntpath.basename(folds_files_dir) + '.txt'
        )
        # Wczytanie wyników dla foldów
        folds_scores = []
        for file_name in folds_files:
            scores = loadDictionary(folds_files_dir, file_name)
            # Przerabianie na numpy
            for field in VECTOR_SCORES:
                scores[field] = np.array(scores[field])
            for field in MATRIX_SCORES:
                scores[field] = np.stack(scores[field], axis=0)
            # Przepisanie wyników
            scores["model_params"]["model_name"] = scores["model_name"]
            scores["model_params"]["fold"]       = scores["fold"]   
            scores["model_params"]["series"]     = series_number 
            scores["model_params"]["trainable_params"] = trainable_params
            # Zapisanie
            folds_scores.append(scores)
        
        
        
        # Sortowanie wg. foldów [0-9]
        folds_scores.sort(key=lambda s: s["fold"])
        if np.array_equal(
                np.array([f["fold"] for f in folds_scores]),
                np.arange(0, EXPECTED_NMBER_OF_FOLDS)) == False:
            printWarning(f'Stacking folds failed. Wrong folds at "{folds_files_dir}"!')
            printWarning(f'Found folds: {str(np.array([f["fold"] for f in folds_scores]))}')
            printWarning(f'Skipping...')
            continue
        
        if stack_folds == False:
            # Dopisanie wyników z pojedyńczych foldów
            models_scores.append(folds_scores)
        else:
            # Stworzneie zbiorowego wyniku
            all_scores = {}
            
            # Jedno wierszowe wyniki
            for field in VECTOR_SCORES:
                temp_stack = []
                for scores in folds_scores:
                    temp_stack.append(scores[field])
                all_scores[field] = np.stack(temp_stack, axis=1)
            # Macierzowe wyniki
            for field in MATRIX_SCORES:
                temp_stack = []
                for scores in folds_scores:
                    temp_stack.append(scores[field])
                all_scores[field] = np.concatenate(temp_stack, axis=1)
                
            # Dopisanie wyniku z wszystkich foldów
            all_scores["model_params"] = folds_scores[0]["model_params"]
            del all_scores["model_params"]["fold"]
            models_scores.append(all_scores)
        
        
    
    ## -------------- FINALNE ZEBRANIE WYNIKÓW DO KUPY -------------- ##
    
    # Stworzenie podsumowania
    printInfo('LOADING AND STACKING SUMMARY...')
    
    summary_scores = []
    if stack_folds == True:
        for scores in models_scores:
            for epoch in range(len(scores['accuracy'])):
                summary = {}
                for field in scores["model_params"].keys():
                    summary[field] = scores["model_params"][field]
                summary["epoch"] = epoch
                for field in VECTOR_SCORES + MATRIX_SCORES:
                    summary[field] = scores[field][epoch]
                summary_scores.append(summary)
    else:
       for folds_scores in models_scores:
           for scores in folds_scores:
                for epoch in range(len(scores['accuracy'])):
                    summary = {}
                    for field in scores["model_params"].keys():
                        summary[field] = scores["model_params"][field]
                    summary["epoch"] = epoch
                    for field in VECTOR_SCORES + MATRIX_SCORES:
                        summary[field] = scores[field][epoch]
                    summary_scores.append(summary)
    
    summary_df = pd.DataFrame.from_records(summary_scores)
    return summary_df















# Główna funkcja trenowania i testowania modeli.
def evaluateModel(params={}): 
    if "verbose" not in params:
        params["verbose"] = 0
    if "flatten" not in params:
        params["flatten"] = False
    if "save_models" not in params:
        params["save_models"] = False
    if "series" not in params:
        params["series"] = 1
    
    ## -------------- WSTĘPNE PRZYGOTOWANIA -------------- ##
    
    params['src_path'] = 'src'
    params['out_path'] = 'out'
    
    # Przygotowanie głównego folderu na wyniki
    main_out_path = os.path.join(params["out_path"], params['classifier_name'], f'SERIES {params["series"]}')
    os.makedirs(main_out_path, exist_ok=True)
    
    # Dodanie opisu konwencji nazw folderów
    classifiers_about = f'FILE NAME PARTS:\n- {params["classifier_name"]},\n'
    for name_part in sorted(params["model_params"].keys()):
        if name_part == 'max_epochs':
            continue
        classifiers_about = classifiers_about + f'- {name_part}  <  {str(params["model_params"][name_part])},\n'
    saveText(main_out_path, params['classifier_name'] + "_INFO.txt", classifiers_about)

    # Ostatnio wczytany plik z danymi
    data = {}
    input_name = None
    



    ## -------------- GENEROWANIE WYNIKÓW DLA WSZYSTKICH FOLDOW -------------- ##

    # Wyznaczenie permutacji parametrow
    params_grid = list(ParameterGrid(params["model_params"]))
    
    # Zamiana "row_limit" zawsze był i by był None dla MELS i CQT
    for i, v in enumerate(params_grid):
        if params_grid[i]["input_name"] in ["mels", "cqt"] or "row_limit" not in params_grid[i]:
            params_grid[i]["row_limit"] = None 
    # Usuwanie duplikatów
    params_grid = [dict(t) for t in {tuple(d.items()) for d in params_grid}]
    # Posortowanie po typie inputu (by często nie ładowac nowych danych)
    params_grid.sort(key=lambda v: v["input_name"])     
        
    # Sprawdzenie wszystkich kombinacji parametróW
    for mpi, model_params in enumerate(params_grid):        
        # Do konsolki...
        printHeader(f'NEXT PARAMETERS, T: {datetime.now().strftime("%H:%M:%S")}')
        printInfo(f'PARAMETER SET {mpi+1}/{len(params_grid)}')
        pprint(model_params)
        print()
        
        # Wczytanie danych tylko jeżeli trzeba...
        if(input_name != model_params["input_name"]):
            input_name = model_params["input_name"]
            print(f'LOADING DATA ({input_name})...')
            data = loadDictionary(params["src_path"], input_name + '_train_data.npy')
        
        # Wyciągniecie najważniejszych danych...
        x = data['x']
        y = data['y']
        
        # Croppowanie danych jeżeli potrzeba...
        if("row_limit" in model_params.keys()):
            if(model_params["row_limit"] != None):
                print(f'LIMITING X TO {model_params["row_limit"]} ROWS...')
                x = x[:,0:model_params["row_limit"],:]
                
        # Spłaszczanie danych jeżeli potrzeba...
        if(params["flatten"]):
            print(f'FLATTENING X...')
            x = x.reshape([x.shape[0], x.shape[1]*x.shape[2]])           
        else:
            # Taki format potrzebuje tensorflow dla CNN:
            print(f'TENSORFLOWING X...')
            x = x.reshape([x.shape[0], x.shape[1], x.shape[2], 1])
            
        print()
            
        # Przygotowanie głównego tekstowego identfikatora modelu i hiperparametrów
        model_type_name = params['classifier_name']
        for model_param_name in model_params.keys():
            # Limit epok nie wchodzi do nazwy modelu
            if model_param_name == 'max_epochs':
                continue
            model_type_name += "_" + str(model_params[model_param_name])
            
        # Przygotowanie folderu na wyniki
        out_path = os.path.join(main_out_path, model_type_name)
        os.makedirs(out_path, exist_ok=True)
    
        # Walidacja krzyżowa
        for fold, fold_indices in enumerate(data['folds']):
            
            # Przygotowanie szczegółowego tekstowego identfikatora modelu i hiperparametrów
            model_name = model_type_name + " - fold " + str(fold+1)
            
            # Sprawdzenie czy ta kombinacja była już liczona
            old_scores = None
            old_model  = None
            file_name     = model_name + '.npy'
            h5_file_name  = model_name + '.h5'
            model_summary_file_name = model_type_name + '.txt'
            if os.path.isfile(os.path.join(out_path, file_name)):
                if "max_epochs" not in model_params:
                    print(f"FILE '{file_name}' WAS FOUND, SKIPPING...")
                    continue
                else:
                    # Sprawdzenie czy wykonano już mniej lub tyle samo epok
                    old_scores = loadDictionary(out_path, file_name)
                    if model_params["max_epochs"] <= old_scores["model_params"]["max_epochs"]:
                        print(f"FILE '{file_name}' WAS FOUND, SKIPPING...")
                        continue
                    # Nowe parametry są takie jak kiedys ale trzeba wykonać więcej epok
                    else:
                        # Sprawdzenie czy zapisano wtedy model
                        if os.path.isfile(os.path.join(out_path, h5_file_name)):
                            print(f'RELOADING OLD MODEL - "{h5_file_name}"...')
                            old_model = tf.keras.models.load_model(os.path.join(out_path, h5_file_name))
                        else:
                            print("WARNING: Needs more epochs but h5 model wasn't found. Retraining...")
                    
            # Wyciągniecie finalnych danych do uczenia i walidacji
            printHeader(f'NEXT FOLD, T: {datetime.now().strftime("%H:%M:%S")}')
            printInfo(f'PARAMETER SET {mpi+1}/{len(params_grid)}')
            printInfo(f'FOLD {fold+1}/{len(data["folds"])}')  
            print()
            
            x_train = x[fold_indices]
            x_test  = x[~fold_indices]
            y_train = y[fold_indices]
            y_test  = y[~fold_indices]
            
            # Przygotowanie modelu z zadanymi hiperparametrami
            if old_model == None:
                print('CREATING MODEL...')
                model = params["create_model_callback"](model_params, x_train.shape[1:], len(data["labels"]))
            else:
                model = old_model
    
            print('TRAINING AND EVALUATING MODEL...\n')
            # Trenowanie modelu. Powinno zwrócić (tablice zawierają epoki):
            #    {
            #         "loss": [],
            #         "train_loss": [],
            #         "accuracy": [],
            #         "train_accuracy": [],
            #         "predictions": [],
            #         "train_time": [],
            #         "test_time": [],
            #    }
            scores = params["train_and_score_model_callback"](
                model, x_train, y_train, x_test, y_test, model_params, old_scores, params["verbose"])  
        
            # Dodanie dodatkowych informacji
            scores["fold"] = fold
            scores["model_name"] = model_type_name
            scores["model_params"] = model_params
            scores["model_params"]["classifier_name"] = params["classifier_name"]

            print(f"\nSAVING FILE '{file_name}'...")
            saveObject(out_path, file_name, scores)
            
            # Sieci neuronowe są zapisywane:
            if "max_epochs" in model_params:
                print(f"SAVING MODEL SUMMARY '{model_summary_file_name}'...")
                with open(os.path.join(out_path, model_summary_file_name),'w') as fh:
                    model.summary(print_fn=lambda x: fh.write(x + '\n'))
                if params["save_models"] == True:
                    print(f"SAVING MODEL '{h5_file_name}'...")
                    model.save(os.path.join(out_path, h5_file_name))
   