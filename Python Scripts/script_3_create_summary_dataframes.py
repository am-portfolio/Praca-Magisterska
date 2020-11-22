import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from pymodules.utilities import *
from pymodules.mlrnhelpers import *
from pymodules.mlevaluator import *





out_path = 'out'
out_main_path = 'out_main'
folders = ['kNN', 'DNN', 'GajhedeCNN', 'HuzaifahCNN', 'SalamonCNN', 'BestCNN']

# Etykiety referencyjne, nazwy etykiet itp.
y_test_data = loadDictionary('src', 'y_test_data.npy')







# Przygotowanie DataFrame podsumującego
def makeSummaryDataFrame(df, predictions = False, trim = True, file_info = False):
    printInfo('GROUPING SCORES...')
    
    # Grupowanie foldów
    df = df.groupby(["model_name", "series", "epoch"])
    groupby_count = len(df)
    groupby_counter = 0
    
    # Przetwarzanie zgrupowanych foldów
    def processFoldGroup(df):
        nonlocal groupby_counter
        groupby_counter += 1
        print(f'Processing groupby: {groupby_counter}/{groupby_count}')
        
        # Zwracany wynik
        series = pd.Series(dtype="object")
        
        if trim == True:
            # Wyrzucenie częsci najlepszych i najgorszych foldów
            TRIM = 2
            df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
            df = df[TRIM:-TRIM]
            
        # Wyciagncięcie referencyjnych etykiet foldów
        labels  = y_test_data['labels']
        y_true  = np.concatenate([y_test_data['y'][i] for i in df.fold])
        if file_info == True:
            y_names = np.concatenate([y_test_data['names'][i] for i in df.fold])
            y_augs  = np.concatenate([y_test_data['augmented'][i] for i in df.fold])
        
        # Przygotowanie danych
        y_prob = None
        STATS_COLS = ["loss", "train_loss", "accuracy", "train_accuracy", "train_time", "test_time"]
        for col in df.columns:
            # Statystyczne wyniki numeryczne
            if col == 'accuracy':
                series['acc'] = df[col].mean()
                series['acc_std'] = df[col].std()
            elif col == 'train_accuracy':
                series['train_acc'] = df[col].mean()
            elif col in STATS_COLS:
                series[col] = df[col].mean()
            # Prawdopodobienstwa
            elif col == 'predictions':
                y_prob = np.vstack(df[col])
            # Foldy są łączone więc ta kolumna jest pomijana
            elif col == 'fold':
                None
            # Własciwosci modelu
            else:
                series[col] = df[col].iloc[0]
    
        # Stworzenie dodatkowych wskaznikow
        series['rci'] = (1 - 1/len(labels))/(1 - series['acc'])
        lacc_scores = class_accuracy_score(y_true, y_prob)
        for i, label in enumerate(labels):
            series[(label+'_acc').lower()] = lacc_scores[i]
            
        # Dodanie prawdopodobieństw
        if predictions == True:
            series['y_prob'] = y_prob
            series['y_true'] = y_true
        if file_info == True:
            series['file_names'] = y_names
            series['file_augs']  = y_augs
        return series
    
    # Przetwarzanie grup i sortowanie
    df = df.apply(processFoldGroup)
    df = df.sort_values('acc', ascending=False).reset_index(drop=True)

    return df






# --------------------- PLIKI SUMMARY ----------------------- #

# Przygotowanie zbiorowych dataframe-ów
with open(os.path.join(out_main_path, 'Train summary.txt'), "w") as text_file:
    def FILE_PRINT(*args, **kwargs):
        print(*args, file=text_file, **kwargs)
        
    for folder in folders: 
        printHeader(folder)
        
        # Sprawdzenie czy już wygenerowano
        file_name = folder + '.summary'
        # if os.path.isfile(os.path.join(out_path, file_name)):
        #    print(f"FILE '{file_name}' WAS FOUND, SKIPPING...")
        #    continue
        
        # Wczytanie danych
        df = collectDataframe(os.path.join(out_path, folder))
        
        # Statystki
        FILE_PRINT(folder)
        # Liczba przetestowanych modeli
        df2 = df.drop_duplicates(["model_name"])
        FILE_PRINT(f'\tUnique models: {len(df2)}')
        # Sumaryczny czas uczenia modeli
        df2 = df.sort_values('epoch', ascending=False)
        df2 = df2.drop_duplicates(["model_name", "series", "fold"])
        FILE_PRINT(f'\tTrained models: {len(df2)}')
        # Sumaryczny czas uczenia modeli
        FILE_PRINT(f'\tTotal train time: {np.round(df2["train_time"].sum()/60/60, 2)} h')
        
        # Tworzenie podsumowania
        df2 = makeSummaryDataFrame(df)
        saveDataFrame(out_path, file_name, df2)
    
    
# Łączenie wszystkiego w jedno
dfs = []
for folder in folders: 
    file_name = folder + '.summary'
    dfs.append(loadDataFrame(out_path, file_name))
df = pd.concat(dfs)
df = df.sort_values('acc', ascending=False).reset_index(drop=True)
saveDataFrame(out_main_path, 'all.summary', df)





# --------------------- PRZYGOTOWANIE FILTRÓW ----------------------- #

# Najlepsze wyniki dla każdego modelu
df = df.sort_values('acc', ascending=False).reset_index(drop=True)
df = df.drop_duplicates(["model_name", "series"]).reset_index(drop=True)

# Najlepsze 50 modeli
df2 = df[df.classifier_name != 'kNN']
ALL_NETWORKS = df2[["model_name", "epoch", "series"]]

# Najlepsze wyniki dla każdego rodzaju klasyfikatora
# Rozgraniczane są klasyfikatory dla różnej liczby epok
df2 = df.drop_duplicates("classifier_name").reset_index(drop=True)
MODELS_BY_BEST_CLASSIFIER = df2[["model_name", "epoch", "series"]]
MODELS_BY_BEST_CLASSIFIER.insert(2, 'bestby', df2["classifier_name"])

# Najlepsze wyniki dla każdego inputu
# (zawsze wyszło SalamonCNN lub BestCNN, ale lepiej porównać tylko Salamon)
df2 = df[df.classifier_name == 'SalamonCNN']
df2 = df2.drop_duplicates("input_name").reset_index(drop=True)
df2 = df2[~df2.input_name.isin(['mfccmax', 'mfccl2'])]
MODELS_BY_BEST_INPUT = df2[["model_name", "epoch", "series"]]
MODELS_BY_BEST_INPUT.insert(2, 'bestby', df2["input_name"])

# Najlepsze wyniki dla każdej etykiety
df2 = df[df.acc > 0.5] # by nie uwzględniać tych zbyt faworyzujących
label_scores = [(label+'_acc').lower() for label in y_test_data['labels']]
dfs = []
for col in label_scores:
    dfs.append(df2.sort_values(col, ascending=False).head(1))
df2 = pd.concat(dfs)
df2.insert(0, 'bestby', label_scores)
df2 = df2.sort_values('acc', ascending=False).reset_index(drop=True)
df2["bestby"] = df2["bestby"].apply(lambda s: s[:-4])
MODELS_BY_BEST_LACC = df2[["model_name", "epoch", "series", "bestby"]]




# --------------------- ZEBRANIE NAJWAŻNIEJSZYCH WYNIKÓW ----------------------- #

FILTERS = {
    'all_networks': ALL_NETWORKS,
    'best_by_classifier': MODELS_BY_BEST_CLASSIFIER,
    'best_by_label': MODELS_BY_BEST_LACC,
    'best_by_input': MODELS_BY_BEST_INPUT,
}
ALL_DFS = {}

# Zbieranie wyników
for folder in folders: 
     printHeader(folder)     
     # Wczytanie danych
     df = collectDataframe(os.path.join(out_path, folder))
     # Zachowanie danych wg filtrów
     for key in FILTERS.keys():
         if key not in ALL_DFS:
             ALL_DFS[key] = []
         for i, row in FILTERS[key].iterrows():
             df2 = df[(df.model_name == row.model_name) & (df.epoch == row.epoch) & (df.series == row.series)]
             if key != 'all_networks':
                 df2["bestby"] = row.bestby
             ALL_DFS[key].append(df2)

# Łączenie wyników
for key in FILTERS.keys():
    ALL_DFS[key] = pd.concat(ALL_DFS[key])

# Przetwarzanie wyników i zapis
for key in FILTERS.keys():
    out_df = ALL_DFS[key]
    
    if key == 'all_networks':
        out_df = makeSummaryDataFrame(out_df, True, False, True)
        saveDataFrame(out_main_path, key + '.summary', out_df)
        continue
    
    # Wyniki z podziałem na foldy
    saveDataFrame(out_main_path, key + '.df', out_df)
    # Wyniki z urednionymi foldami
    out_df = makeSummaryDataFrame(out_df, True, False)
    saveDataFrame(out_main_path, key + '.summary', out_df)
        
    # Główne wykresy
    for i, row in FILTERS[key].iterrows():
        series = out_df[(out_df.model_name == row.model_name) & (out_df.epoch == row.epoch) & (out_df.series == row.series)].iloc[0]
        
        # Zebranie danych
        y_true = series["y_true"]
        y_prob = series["y_prob"]
        model_name = series["model_name"]
        classifier_name = series["classifier_name"]
        if key != 'best_by_classifier':
            bestby = 'BestBy' + row.bestby.upper()
        else:
            bestby = 'BestBy' + row.bestby
            
        print('PLOTS FOR: ', bestby, model_name)
        ratePredictions(
            y_true, y_prob,
            y_test_data["labels"],
            bestby,
            os.path.join(out_main_path, key, f'[{bestby}] {model_name}')
        )
        with open(os.path.join(out_main_path, key, f'[{bestby}] Model params.txt'), "w") as text_file:
            for index, value in series.items():
                print(f"{index}:\n\t{value}", file=text_file)
                
    # Tablice pomyłek VS
    for i, row1 in FILTERS[key].iterrows():
        series1 = out_df[(out_df.model_name == row1.model_name) & (out_df.epoch == row1.epoch) & (out_df.series == row1.series)].iloc[0]
        for j, row2 in FILTERS[key].iterrows():
            if i == j:
                continue
            series2 = out_df[(out_df.model_name == row2.model_name) & (out_df.epoch == row2.epoch) & (out_df.series == row2.series)].iloc[0]
            
            y_true1 = series1["y_true"]
            y_prob1 = series1["y_prob"]
            y_pred1 = np.argmax(y_prob1, axis=1)
            cnf1 = sklearn.metrics.confusion_matrix(y_true1, y_pred1, normalize='true')
            y_true2 = series2["y_true"]
            y_prob2 = series2["y_prob"]
            y_pred2 = np.argmax(y_prob2, axis=1)
            cnf2 = sklearn.metrics.confusion_matrix(y_true2, y_pred2, normalize='true')
            labels = y_test_data["labels"]
            if key != 'best_by_classifier':
                bestby1 = 'BestBy' + row1.bestby.upper()
                bestby2 = 'BestBy' + row2.bestby.upper()
            else:
                bestby1 = 'BestBy' + row1.bestby
                bestby2 = 'BestBy' + row2.bestby
            
            save_to = os.path.join(out_main_path, key, f'A vs B Confusion Matrices')
            
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
            plt.title(f'{bestby2} → {bestby1}')
            for _ in range(4):
                plt.tight_layout()
            os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(save_to, f'{bestby2}_vs_{bestby1}_ConfusionMatrix.png'))
                
                
            
            
            
            



    