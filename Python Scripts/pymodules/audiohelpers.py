import config
from .utilities import *
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


# Interpoluje dane audio zmieniając w efekcie długosc i częstotliwosc
def interpSample(audio, semitones=1):
    ratio = 2**(-semitones/12)
    return np.interp(np.linspace(0,1, int(len(audio)*ratio)), np.linspace(0,1,len(audio)), audio)
    
# Mapuje fade na sin
def fadeValue(v):
    return np.sin(v*0.5*np.pi)**2
# Smart fadeout - czas sciszenia to 2-ktotnosc ostatniego okresu
def addSmartFade(audio):
    count = 0
    sign = np.sign(audio[-1])
    first_sign = sign
    for i in range(len(audio)):
        if (np.sign(audio[-i-1]) == sign):
            count = count + 1
        else:
            sign = -sign
            if (sign == first_sign):
                break;
    fadeout = np.min([len(audio), count*2])
    for i in range(len(audio) -1, len(audio) - 1 - fadeout, -1):
        audio[i] = audio[i] * fadeValue((len(audio) - 1 - i)/fadeout)
    return audio

# Przycina/wydłuża dźwięk tak by miał okresloną długosc
def changeLength(audio, length = config.length_in_samples, fadeout = config.fadeout_in_samples):
    # Przycięcie/wydłuzenie pliku do zadanej długości
    if(len(audio) > length):
        audio = audio[0:length]
    else:
        if(audio[-1]>0.0005):
            audio = addSmartFade(audio)
        audio = np.pad(audio, (0, length - len(audio)))
    # Dodanie fadeout'u
    for i in range(len(audio) -1, len(audio) - 1 - fadeout, -1):
        audio[i] = audio[i] * fadeValue((len(audio) - 1 - i)/fadeout)
    return audio

# Ucina ciszę na początku serii sampli, fadein w samplach.
def trimInSilence(audio, treshold = config.amplitude_treshold, fadein = config.fadein):
    above_treshold_id = 0
    for i in range(len(audio)):
        if(audio[i] > treshold):
            above_treshold_id = i
            break
    if(above_treshold_id > fadein):
        new_start_id = above_treshold_id - fadein
        audio = audio[new_start_id:]
        for i in range(fadein):
            audio[i] = audio[i] * (i / fadein)
    return audio

# Wczytuje plik w zadanym próbkowaniu i usuwa ewentualną ciszę (z zadanym progiem) na początku.
def loadSample(file, sample_rate = config.sample_rate):
    # Wczytanie plikow w zadanym próbkowaniu
    audio, sr = librosa.load(file, sr=sample_rate)
    # Normalizacja zakresu
    audio = librosa.util.normalize(audio)
    #Przycięcie szumu na początku pliku
    audio = trimInSilence(audio)
    # Zwraca dane, próbkowanie i serię czasów
    times = np.linspace(0, len(audio)/sr, len(audio))
    return audio, sr, times


# Wyznacza krzywą RMS określajacą głoścność na podstawie spektogramu
# Dla relative=True zwróci wynik w zakresie 0-1.
# Jeśli podany jest czas to przeprowadzi interpolację pod chwile czasu.
def calculateRMS(audio, time=None, n_fft=1024, relative=True):
    S = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=n_fft//4, center=False)
    S_amp = np.abs(S)
    rms = np.sqrt(np.sum(S_amp**2, axis=0))
    if relative:
        rms = normalize(rms)
    if time is not None:
        rmstime = np.linspace(0, time[-1], len(rms))
        rms = np.interp(time, rmstime, rms)  
    return rms


# Sprowadza plik do standardowej długości i samplerate i wyznacza jego realną długość.
def calcuateRealLength(file, tresholds = [0.25, 0.1, 0.05, 0.01], plot=False):
    # Wczytanie pliku i wyznaczenie RMS
    audio, sr, time = loadSample(file)
    if len(audio) < 1024:
        audio = np.pad(audio, (0, 1024 - len(audio)))
    rms = calculateRMS(audio, time)

    # Wyznaczenie "odczuwalnej" długości sampla
    real_lengths = []
    for treshold in tresholds:
        real_length = time[-1]
        end_index = 0
        for i in range(len(rms) - 1, -1, -1):
            if(rms[i] > treshold):
                end_index = i
                break
        if(end_index != 0):
            real_length = time[end_index]
        real_lengths.append(real_length)

    # Opcionalne plotowanie
    if plot:
        f = plt.figure(figsize=(7, 2.5))
        f.add_subplot(111)
        plt.plot(time, audio, label='Sygnał', color='grey', alpha=0.6)
        plt.plot(time, rms, label='RMS', color=getTabColor(0))
        for i, treshold in enumerate(tresholds):
            plt.vlines(real_lengths[i], -1, 1, color=getTabColor(i+1), linestyle='--', label=f'RMS < {treshold}')

        plt.legend(title=os.path.basename(file), loc="lower left")
        plt.ylabel('y / RMS')
        plt.xlabel('Czas [s]')
        plt.tight_layout()

    return real_lengths










sr = config.sample_rate
n_fft       = 1024
hop_length  = 512
cqt_bpo     = 12
cqt_bins    = 108  # max=ok. 16744
cqt_fmin    = None # ok. 32.70 C1
mels_fmax   = 440*2**((-45+cqt_bins)/12) # ok 16744.036179238312
mels_fmin   = 0
mels_n      = 108
window      ='hann'


# Normalizacja by maksymlna wartoć bezwzględna macierzy lub wierzy była równa 1
def maxNormalizeS(S,rowWise=False):
    if(rowWise):
        for i in range(S.shape[0]):
            v = np.max(np.abs(S[i,:]))
            if(v > 0):
                S[i,:] = S[i,:] / v 
    else:
        v = np.max(np.abs(S))
        if(v > 0):
            S = S / v 
    return S

# Standardowa normalizacja całej macierzy do zakresu [0,1]
def minmaxNormalizeS(S):
    S = S - np.min(S)
    v = np.max(S)
    if(v > 0):
        S = S / v 
    return S

# Normalizacja by norma L2 całej macierzy lub weirzy była równa 1
def normNormalizeS(S,rowWise=False):
    if(rowWise):
        for i in range(S.shape[0]):
            norm = np.linalg.norm(S[i,:], ord=2)
            if(norm > 0):
                S[i,:] = S[i,:] / norm 
    else:
        norm = np.linalg.norm(S, ord=2)
        if(norm > 0):
            S = S / norm 
    return S


# Generuje wybrane do pracy MEL-S
def audioToMELS(audio, norm=True):
    # Ustawienie stałej długosci
    audio = changeLength(audio)
    # Generowanie
    S = librosa.feature.melspectrogram(
        audio, sr=sr,
        n_mels=mels_n, fmax=mels_fmax, fmin=mels_fmin,
        n_fft=n_fft, hop_length=hop_length, window = window,
        center=False, dtype=np.complex128
    )
    S = np.abs(S)
    S = librosa.power_to_db(S, ref=np.max)
    S = S[:,0:86] # Ogranicznie kolumn (44100/512 to ok 86)
    if(norm):
        S = minmaxNormalizeS(S)
    return S
# Wyswietla wybrnay do pracy MELS
def showMELS(S):
    librosa.display.specshow(
        S, sr=sr, x_axis='time', y_axis='mel',
        fmax=mels_fmax, fmin=mels_fmin,
        hop_length=hop_length)
    plt.xlabel('t [s]')
    plt.ylabel('f [Hz]')
    plt.xticks([0, 0.3, 0.6, 0.9])
    plt.title(f'MEL-S\n[{S.shape[0]},{S.shape[1]}] ({S.shape[0]*S.shape[1]})')


# Generuje wybrane do pracy CQT
def audioToCQT(audio, norm=True):
    # Ustawienie stałej długosci
    audio = changeLength(audio)
    S = librosa.cqt(
        audio, sr=sr,
        n_bins=cqt_bins, fmin=cqt_fmin, bins_per_octave=cqt_bpo,
        hop_length=hop_length, window = window,
    )
    S = S[:,0:86] # Ogranicznie kolumn (44100/512 to ok 86)
    S = librosa.amplitude_to_db(np.abs(S), ref=np.max)   
    if(norm):
        S = minmaxNormalizeS(S)
    return S
# Wyswietla wybrnay do pracy MELS
def showCQT(S, showAxis=True):
    librosa.display.specshow(
        S, sr=sr, x_axis='time', y_axis='cqt_hz',
        bins_per_octave=cqt_bpo, fmin=cqt_fmin,
        hop_length=hop_length)
    if(showAxis):
        plt.xlabel('t [s]')
        plt.ylabel('f [Hz]')
        plt.xticks([0, 0.3, 0.6, 0.9])
        plt.title(f'CQT\n[{S.shape[0]},{S.shape[1]}] ({S.shape[0]*S.shape[1]})')
    else:
        plt.xlabel(None)
        plt.ylabel(None)
        plt.xticks([])
        plt.yticks([])


# Generuje wybrane do pracy MFCC
def audioToMFCC(audio):
    S = librosa.feature.mfcc(S=audioToMELS(audio, False), n_mfcc=None)
    return maxNormalizeS(S)

def audioToCQCC(audio):
    S = librosa.feature.mfcc(S=audioToCQT(audio, False), n_mfcc=None)
    return maxNormalizeS(S)

# Generuje wybrane do pracy MFCC + wiersze normalizowane L2
def audioToMFCCL2(audio):
    S = librosa.feature.mfcc(S=audioToMELS(audio, False), n_mfcc=None)
    return normNormalizeS(S, rowWise=True)

# Generuje wybrane do pracy MFCC + wiersze normalizowane max
def audioToMFCCMAX(audio):
    S = librosa.feature.mfcc(S=audioToMELS(audio, False), n_mfcc=None)
    return maxNormalizeS(S, rowWise=True)

# Wyswietla wybrnay do pracy MFCC
def showMFCC(S, name='MFCC'):
    librosa.display.specshow(
        S, sr=sr, x_axis='time',
        hop_length=hop_length,
        cmap='coolwarm_r'
    )
    plt.xlabel('t [s]')
    plt.xticks([0, 0.3, 0.6, 0.9])
    plt.title(f'{name}\n[{S.shape[0]},{S.shape[1]}] ({S.shape[0]*S.shape[1]})')
    
    
    
