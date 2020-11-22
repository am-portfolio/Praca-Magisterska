"""
    FUNKCJE POMOCNICZE DO MASHINE LEARNINGU,
    np. wyznaczanie confusion matrix i generownaie wykresów
"""

import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
import json
from operator import add

from .utilities import *

# Zwraca dokładnosc klasyfikacji dla każdej klasy osobno
def class_accuracy_score(y_true, y_pred, is_prob = True):
    if is_prob == True:
        y_pred = np.argmax(y_pred, axis=1)
    return np.diag(sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true'))

# Wyznacza różne wzkaźniki jakosci (z podziałem na klasy) na podstawie ID klasy własciwej i przewidzianej.
def mlrnMetrics(y_true, y_pred):
    ACC = sklearn.metrics.accuracy_score(y_true, y_pred) # np.mean(y_true == y_pred)
    
    # TABLICA POMYŁEK (taka że: x=predicted i y=actual)
    cnf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Wersja znormalizowana tak że suma w każdym wierszu to 1 (nieweluje wpływ różnicy liczby próbek różnych klas)
    cnf_matrix_norm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')

    # WPOZOSTAŁE METRYKI DLA KAZDEJ KLASY OSOBNO
    # Liczone przy założeniu: x=predicted, y=actual
    # https://stackoverflow.com/questions/50666091/true-positive-rate-and-false-positive-rate-tpr-fpr-for-multi-class-data-in-py
    
    # True Positives - Wartosci na przekątnej tablicy
    TP = np.diag(cnf_matrix)
    # False Positives - Suma w danej kolumnie bez tego co było na przekątnej (TP)
    FP = cnf_matrix.sum(axis=0) - TP
    # False Negatives - Suma w danym wierszu bez tego co było na przekątnej (TP)
    FN = cnf_matrix.sum(axis=1) - TP
    # True Negatives - Cała reszta
    TN = cnf_matrix.sum() - (FP + FN +TP)

    return {
        # Tablica pomyłek i tablica pomyłek znormalizowana/zbalansowana (niweluje różny rozmiar klas)
        'cnf_matrix': cnf_matrix,
        'cnf_matrix_norm': cnf_matrix_norm,
        
        # Podstawowe rzeczy typu True Positives, True Negatives...
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'tn': TN,
        
        # Accuracy - Jaką częsc próbek przypisano do odpowiedniej klasy
        'acc': ACC,
        
        # Random Classifier Improvement - jak sie poprawiło wzgledem losowania
        'rci': (1 - 1/len(TP))/(1 - ACC),
        
        # Balanced Accuracy - To co wyżej ale każda klasa ma wpływ na taką samą ilosc procentow (niweluje rozmiary)
        'bacc': sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
        
        # Prediction Accuracy - Jaka częsc tego co miała być True/False taka była
        'pacc': (TP+TN)/(TP+FP+FN+TN),
        
        # True Positive Rate (Sensitivity / Recall) - Jaka częsc z tych co miała byc True była True
        'tpr': TP/(TP+FN),
        # True Negative Rate (Specificity) - Jaka częsc z tych co miała byc Falsa była False
        'tnr': TN/(TN+FP),
        # False Negative Rate (1 - Sensitivity) - Jaka częsc z tych co miała byc True była False
        'fnr': FN/(TP+FN),
        # False Positive Rate (1 - Specificity) - Jaka częsc z tych co miała byc False była True
        'fpr': FP/(FP+TN),
        
        # Positive Predictive Value (Precision) - Jaka czesc wszystkich True jest poprawna
        'ppv': TP/(TP+FP),
        # Negative Predictive Value - Jaka czesc wszystkich False jest poprawna
        'npv': TN/(TN+FN),
        
        # False Discovery Rate (1 - Precision) - Jaka czesc wszystkich True jest NIEpoprawna
        'fdr': FP/(TP+FP),
        # False Omision Rate (1 - NPV) - Jaka czesc wszystkich True jest NIEpoprawna
        'for': FN/(TN+FN),
    }


# Przyjmuje wynik z mlrnMetrics i przedstawia confusion matrix jako heatmap
def plotConfusionMatrix(mlrn_metrics, labels, model_name = 'UNTITLED', save_to = None):
    for cnf_matrix in [mlrn_metrics['cnf_matrix'], mlrn_metrics['cnf_matrix_norm']]:
        # Formatowanie wartosci:
        fmt = '0.1f'
        is_normalized = True
        cnfm = cnf_matrix * 1
        if(issubclass(cnfm.dtype.type, np.integer)):
            fmt = 'd'
            is_normalized = False
        else:
            cnfm  = cnfm * 100
        # Wykres:
        plt.figure(figsize=(5,5))
        cnf_matrix_df = pd.DataFrame(cnfm, index=labels, columns=labels)
        sn.heatmap(cnf_matrix_df, annot=True, fmt=fmt, cbar=False, square=True, cmap="cubehelix_r")
        plt.xlabel('Predykcja')
        plt.ylabel('Poprawna etykieta')
        plt.gca().xaxis.tick_top()
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.xticks(rotation=45) 
        plt.title(f'{model_name} {" (Normalized)" if is_normalized else ""}')
        for _ in range(4):
            plt.tight_layout()
        if(save_to):
            os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(save_to, f'{model_name}_ConfusionMatrix{"Norm" if is_normalized else "V"}.png'))
            
    
# Przyjmuje wynik z mlrnMetrics i przedstawia confusion matrix w postaci barów
def plotConfusionBars(mlrn_metrics, labels, model_name = 'UNNAMED', save_to = None):
    for log in [True, False]:
        # Wyznaczenie rozkładu prawdopodobieństwa wybrania każdej klasy (już po argmax).
        pred_ratios = mlrn_metrics['cnf_matrix_norm']
        class_count = len(labels)
        x_axis = np.arange(0, class_count, 1)
        
        fig = plt.figure(figsize=(10,12))
        fig.suptitle(f'{model_name} (Normalized{", Log10" if log else ""})', fontsize=12, y=0.98)
        for i in range(class_count):
            plt.subplot(np.ceil(class_count/2), 2, i+1)
            y = pred_ratios[i]
            bar = plt.bar(x_axis, y, width=0.9, log=log, color='tab:red')
            y_min = np.min(y)
            y_max = np.max(y)
            y_pad = (y_max - y_min)/4
            if log:
                plt.ylim(1/900,1)
            else:
                plt.ylim(np.max([0, y_min - y_pad]), np.min([1, y_max + y_pad]))
            bar[i].set_color('tab:green')
            
            # W tych po lewej...
            if(log):
                plt.yticks([10**0, 10**-1, 10**-2])
            else:
                plt.yticks([1,0.75,0.5,0.25])

            plt.ylabel('Predykcje')
            plt.xticks(x_axis, labels, rotation=45, fontsize=8)
            plt.xlabel('Etykiety')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        if(save_to):
            os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(save_to, f'{model_name}_ConfusionBars{"Log10" if log else "V"}.png')) 
    
   
# Przyjmuje wynik z mlrnMetrics i przedstawia różne wartosci w postaci barów
def plotMlrnMetricsBar(mlrn_metrics, labels, model_name = 'UNNAMED', save_to = None):
    class_count = len(labels)
    x_axis = np.arange(0, class_count, 1)
    
    short_names = ['tpr', 'tnr', 'ppv',
                   'fnr', 'fpr', 'fdr']
    full_names  = ['Czułość (TPR)' , 'Specyficzność (TNR)', 'Precyzja (PPV)',
                   '1 - Czułość (FNR)', '1 - Specyficzność (FPR)', '1 - Precyzja (FDR)']
    fig = plt.figure(figsize=(7,4))
    fig.suptitle(f'{model_name} per-class metrics', fontsize=12)
    for i in range(len(short_names)):
        plt.subplot(2, 3, i+1)
        y = mlrn_metrics[short_names[i]]
        plt.bar(x_axis, y, width=0.9, color='tab:green' if i < 3 else 'tab:red')
        plt.title(full_names[i])
        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        y_pad = (y_max - y_min)/4
        plt.ylim(np.max([0, y_min - y_pad]), np.min([1, y_max + y_pad]))
        plt.xticks(x_axis, labels, rotation=60, fontsize=6)
        for j, tf in enumerate(np.isnan(y)):
            if tf == True:
                plt.scatter(j, (y_min + y_max)/2, marker='X', color='black')
                
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    if(save_to):
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, f'{model_name}_MlrnMetricsBar.png'))
   
    
# Przyjmuje wynik z mlrnMetrics i przedstawia rózne wartosci w postaci tabeli
def plotMlrnMetricsTable(mlrn_metrics, labels, model_name = 'UNNAMED', save_to = None):
    class_count = len(labels)

    _metrics = [('tpr' ,'Czułość (TPR)'), ('tnr' ,'Specyficzność (TNR)'), ('ppv', 'Precyzja (PPV)'),
                ('npv', 'NPV'), ('pacc', 'PACC'),
            ('fnr', '1 - Czułość (FNR)'), ('fpr', '1 - Specyficzność (FPR)'), ('fdr', '1 - Precyzja (FDR)')]
    short_names = list(zip(*_metrics))[0]
    full_names  = list(zip(*_metrics))[1]
    rowColours  = ['lightgreen']*5 + ['lightcoral']*3
    
    data  = [mlrn_metrics.get(k) for k in short_names]
    cells = np.round(data, decimals=3)
    
    # Surowe dane
    fig, ax = plt.subplots(figsize=(8,2.5))
    fig.suptitle(f'{model_name} per-class metrics (ACC' + 
                 f': {np.round(mlrn_metrics["acc"], decimals=5)})', fontsize=12, y=0.93)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cells, loc='center',rowLabels=full_names, colLabels=labels,
             rowColours=rowColours, colColours=['lightgray']*len(labels))
    fig.tight_layout()
    if(save_to):
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, f'{model_name}_MlrnMetricsTableV.png'))
        
    # Dane w postaci statystycznej
    MMMlabeles = ['Mean', 'Median', 'Stdv.', 'Min.', 'Max.']
    MMMcells = np.round(np.transpose([np.mean(data, axis=1), np.median(data, axis=1),
               np.std(data, axis=1), np.min(data, axis=1), np.max(data, axis=1)]), decimals=3)
    MMMcells = MMMcells.astype('str')
    ArgMin = np.argmin(data, axis=1)
    ArgMax = np.argmax(data, axis=1)
    MMMcells[:,3] = [f'{v} ({labels[ArgMin[i]]})' for i, v in enumerate(MMMcells[:,3]) ]
    MMMcells[:,4] = [f'{v} ({labels[ArgMax[i]]})' for i, v in enumerate(MMMcells[:,4]) ]
    fig, ax = plt.subplots(figsize=(8,2.5))
    fig.suptitle(f'{model_name} per-class metrics (ACC' + 
                 f': {np.round(mlrn_metrics["acc"], decimals=5)})', fontsize=12, y=0.93)
    ax.axis('off')
    ax.axis('tight')
    ax.table(MMMcells, loc='center', rowLabels=full_names, colLabels=MMMlabeles,
             rowColours=rowColours, colColours=['lightgray']*len(labels))
    fig.tight_layout()
    if(save_to):
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(os.path.join(save_to, f'{model_name}_MlrnMetricsTableStats.png'))

       
 # Przyjmuje trzywymairową posortowaną tablicę taką że pierwszy wymiar opisuje poprawną kasę,
# drugi to jedna predykcja, a trzeci to prawdpodpowieństwa każdej klas w danej predykcji.
def plotProbBoxplots(y_pred_split, labels, model_name = 'UNNAMED', save_to = None):
    for i in range(len(y_pred_split)):
        data = y_pred_split[i]
        plt.figure(figsize=(6,3))
        plt.title(f'[{model_name}] prob. dist. ({labels[i]})')
        plt.subplots_adjust(bottom=0.15)
        # Boxplot
        box = plt.boxplot(data, labels=labels, whis=[5,95],
                    flierprops=dict(markeredgecolor='gray', alpha=0.5, marker='.')) 
        plt.xticks([])
        plt.ylabel('Prawdopodobieństwo')
        # Tabelka pod boxplotem
        MMMlabeles = ['Avg.', 'Med.', 'Stdv.', 'Min.', 'Max.']
        MMMcells = np.round([np.mean(data, axis=0), np.median(data, axis=0),
                   np.std(data, axis=0), np.min(data, axis=0), np.max(data, axis=0)], decimals=3)
        colColours = ['lightcoral']*len(labels)
        colColours[i] = 'lightgreen'
        tbl = plt.table(cellText=MMMcells, rowLabels=MMMlabeles, colLabels=labels, loc='bottom',
                        colColours=colColours, rowColours=['lightgray']*len(MMMlabeles))
        tbl.scale(1, 1.5)
        for _ in range(6): # Za każdym razem daje inny wynik, dla 6 wygląda ładnie
            plt.tight_layout()
        if(save_to):
            os.makedirs(save_to, exist_ok=True)
            plt.savefig(os.path.join(save_to, f'{model_name}_ProbBox{labels[i]}.png'))
    

# Generuje statystyki opisujące stan badanego modelu w oparciu o podane predykcje i poprawne wyniki.
# y_true - indeksy poprawnych etykiet
# y_prob - prawdopodobienstwa etykiet
# labels - Nazwy klas przypisywanych na wykresach zamiast id 0, 1, ...
# model_name - Identyfikator badanego modelu. Np w formacie: TYP_PARAMETR_PROBA_EPOKA
# save_to - Miejsce w jakie nalerzy zapisać wyniki
def ratePredictions(y_true, y_prob, labels=None, model_name = 'UNTITLED', save_to = None, close = True):
    # Przygotowanie predykcji w postaci indeksów
    y_pred = np.argmax(y_prob, axis=1)
    
    # Sortowanie tablic po prawdziwych klasach
    sort_order = y_true.argsort()
    y_prob = y_prob[sort_order]
    y_true = y_true[sort_order]
    y_pred = y_pred[sort_order]
    
    # Podzielenie wyników predykcji ze względu na prawdziwą klasę
    _, split_points = np.unique(y_true, return_index=True)
    y_prob_split = np.split(y_prob, split_points[1:], axis=0)
    
    # Wyznaczenie różnych podstawowych metryk maszynowego uczenia na podstawie samych labeli
    mlrn_metrics = mlrnMetrics(y_true, y_pred)
    
    # Fallback: opis elementów na osi wg indeksu klasy:
    num_labels = np.arange(0, len(split_points), 1)
    if(labels is None):
        labels = num_labels
          
    # === WYKRESY === #
        
    # Plotowanie tablicy pomyłek.
    plotConfusionMatrix(mlrn_metrics, labels, model_name, save_to)
    # Plotowanie tablicy pomyłek w postaci bar plotów
    plotConfusionBars(mlrn_metrics, labels, model_name, save_to)
    # Plotowanie różnych danych typowych dla machine lerningu w postaci barów i tabeli
    plotMlrnMetricsBar(mlrn_metrics, labels, model_name, save_to)
    plotMlrnMetricsTable(mlrn_metrics, labels, model_name, save_to)
    
    # Plotowanie boxplotów prawdopodobieństw
    plotProbBoxplots(y_prob_split, labels, model_name, save_to)
    if(close):
        plt.close('all')

















# Usuwa pozostałe epoki
def bestEpochOnly(df):
    return df.sort_values('acc', ascending=False).drop_duplicates(["model_name", "series"]).reset_index(drop=True)


# Wyswietlanie zaleznosci jednego 
def plotBy(df, by, name, kind='box', limit=5, astype=None, metric='acc', metric_name = 'ACC'):
    if limit == None:
        limit = 1000
    # Zostawienie najlepszej epoki
    df = bestEpochOnly(df)
    # Przygotowanie danych
    df = df.groupby(by).apply(lambda v: v.nlargest(limit, metric))[metric].reset_index()
    
    ids = df['level_1'].values
    x   = df[by].values
    y   = df[metric].values
    
    # Wartosci na osi x
    if astype != None:
        x = x.astype(astype)
        if astype == 'bool':
            x = ['Tak' if v else 'Nie' for v in x]
    if by == 'input_name':
        x = [s.upper() for s in x]
    elif by == 'dropout':
        x = [f'{str(int(v*50))}%' for v in x]
    
    # Plot
    if kind == 'box':
        df = pd.DataFrame({'x': x, 'y': y}).groupby('x')['y'].apply(list).reset_index()
        x = df['x'].values
        y = df['y'].values
        plt.boxplot(y, labels=x)
    elif kind == 'scatter':
        x = [str(v) for v in x]
        for i in range(len(ids)):
            plt.scatter(x[i], y[i], marker=f'${ids[i] + 1}$', color=getTabColor(ids[i]))
        indexes = np.unique(x, return_index=True)[1]
        x_uq = [x[index] for index in sorted(indexes)]  
        if len(x_uq) <= 3:
           plt.xticks(np.arange(-1, len(x_uq)+1), np.concatenate([[''], x_uq, ['']]))
    else:
        plt.plot(x, np.mean(np.array(y), axis=1), color="tab:blue")
    plt.xlabel(name)
    plt.ylabel(f"$ \overline{{{metric_name}}} $")


# Plotuje epoki dla najlepszych modeli
# Ploty numeruje od najwyższej dokładnosci
def plotEpchs(df, legend = False, MIN_EPOCH=10):
    top_models = bestEpochOnly(df)["model_name"].values
    max_epochs = []
    min_accs = []
    max_accs = []
    for i, model_name in enumerate(top_models):
        df2 = df[df.model_name == model_name]
        df2 = df2.sort_values('epoch', ascending=False)
        plt.plot(df2["epoch"].values, df2["acc"].values, color=getTabColor(i), label=i+1)  
        max_epochs.append(np.max(df2["epoch"].values))
        min_accs.append(df2[df2.epoch >= MIN_EPOCH]["acc"].min())
        max_accs.append(df2["acc"].max())
    plt.xlim([MIN_EPOCH, np.max(max_epochs)])
    plt.ylim([np.min(min_accs), np.max(max_accs)])
    plt.xlabel('Epoka')
    plt.ylabel(f"$ \overline{{ACC}} $")
    if legend == True:
        plt.legend(fontsize='small', ncol=3, loc='lower right')

