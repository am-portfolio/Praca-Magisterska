# SKRYPT GENERUJĄCY RÓŻNE WYKRESY UŻYTE W PRACY

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
import librosa.display
import sounddevice as sd
import time
from tqdm import tqdm
import scipy.fftpack
from scipy import signal
from matplotlib import image
import skimage.measure
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer
import random



def CLOSE_PLOTS():
    plt.close()
    return









#LINREG
N = np.arange(0, 100, 2)
s = 2*np.pi*N*0.33

plt.figure(figsize=(7,2))
plt.subplot(1,3,1)
plt.plot(N,s, label='s=2πrN')
plt.title('Model matematyczny')
plt.xlabel('N')
plt.ylabel('s [m]')
plt.legend()

s2 = s + np.random.normal(5,5,len(s))
plt.subplot(1,3,2)
plt.scatter(N,s2, marker='.', color='tab:orange')
plt.title('Symulowane pomiary')
plt.xlabel('N(t)')
plt.ylabel('s(t) [m]')

regr = LinearRegression()
regr.fit(N.reshape(-1, 1), s2.reshape(-1, 1))
s3 = regr.predict(N.reshape(-1, 1))
plt.subplot(1,3,3)
plt.plot(N,s, label='s=2πrN', color='tab:blue')
plt.plot(N,s3, label='s=aN+b', color='tab:orange')
plt.title('Wynik regresji liniowej')
plt.xlabel('N')
plt.ylabel('s [m]')
plt.legend()


plt.tight_layout()
plt.savefig('plots/lin_reg.png')
CLOSE_PLOTS()













# SAMPLE RATE
plt.figure(figsize=(7,2))

plt.subplot(1,3,1)
plt.title('Fala 1Hz')
t = np.arange(0, 2, 0.01)
y = np.sin(t * np.pi*2)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')

plt.subplot(1,3,2)
plt.title('Próbkowanie: 20Hz')
t2 = np.arange(0, 2, 0.05)
y2 = np.sin(t2 * np.pi*2)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y, color='black', alpha=.1)
plt.scatter(t2,y2, marker='.')
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.subplot(1,3,3)
plt.title('Próbkowanie: 5Hz')
t2 = np.arange(0, 2, 0.2)
y2 = np.sin(t2 * np.pi*2)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y, color='black', alpha=.1)
plt.scatter(t2,y2, marker='.')
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.tight_layout()
plt.savefig('plots/sample_rates.png')
CLOSE_PLOTS()














# BIT DEPTH
plt.figure(figsize=(7,2))

plt.subplot(1,3,1)
plt.title('Fala 1Hz')
t = np.arange(0, 2, 0.01)
y = np.sin(t * np.pi*2)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')

yr = (y + 1)/2
plt.subplot(1,3,2)
plt.title('Rozdzielczość: 3 bity')
y2 = np.round(yr * 7)/7*2 - 1
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y, color='black', alpha=.1)
plt.plot(t,y2)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.subplot(1,3,3)
plt.title('Rozdzielczość: 2 bity')
y2 = np.round(yr * 3)/3*2 - 1
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y, color='black', alpha=.1)
plt.plot(t,y2)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.tight_layout()
plt.savefig('plots/bit_depth.png')
CLOSE_PLOTS()
















# WAVE
plt.figure(figsize=(7,2))

plt.subplot(1,3,1)
plt.title(r'$A=1, f=1 Hz, φ_0=0$')
y = np.sin(t * np.pi*2)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')

plt.subplot(1,3,2)
plt.title(r'$A=2, f=5 Hz, φ_0=0$')
y2 = np.sin(t * np.pi*2*5)*2
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y2)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.subplot(1,3,3)
plt.title(r'$A=5, f=1 Hz, φ_0=\frac{π}{2}$')
t2 = np.arange(0, 2, 0.2)
y2 = np.sin(t * np.pi*2 + np.pi/2)*5
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y2)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.tight_layout()
plt.savefig('plots/frq_amp_phase.png')
CLOSE_PLOTS()















# HARMONIC WAVES
plt.figure(figsize=(7,2))
t = np.arange(0, 2, 0.001)
f = 3

plt.subplot(1,3,1)
y = t*0;
plt.title(r'$A=[1,0,0,\frac{1}{4},0,0,\frac{1}{7}...]$')
for i in range(1, 1000, 3):
    y += np.sin(2*np.pi*t*f*i)/i
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')

plt.subplot(1,3,2)
y = t*0;
plt.title(r'$A=[1,\frac{1}{2},\frac{1}{3},\frac{1}{4},\frac{1}{5}...]$')
for i in range(1, 1000):
    y += np.sin(2*np.pi*t*f*i)/i
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.subplot(1,3,3)
y = t*0;
plt.title(r'$A=[1,0,\frac{1}{3},0,\frac{1}{5},0...]$')
for i in range(1, 1000, 2):
    y += np.sin(2*np.pi*t*f*i)/i
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t,y)
plt.xlim(0,1)
plt.xlabel('t[s]')

plt.tight_layout()
plt.savefig('plots/complex_waves.png')
CLOSE_PLOTS()














# HARMONIC WAVES2
def FFT(t, y):
    n = len(t)
    dt = t[1] - t[0]
    
    f = np.fft.fftfreq(n, dt)
    f = f[:int(n/2)]
    
    a = np.fft.fft(y) * dt  
    a = np.abs(a)
    a = a[:int(n/2)]
    return (f, a)

plt.figure(figsize=(7,2))
t = np.arange(0, 2, 0.001)
f = 3
xlimmax = 50

plt.subplot(1,3,1)
y = t*0;
plt.title(r'$A=[1,0,0,\frac{1}{4},0,0,\frac{1}{7}...]$')
for i in range(1, 1000, 3):
    y += np.sin(2*np.pi*t*f*i)/i
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(0,xlimmax)
plt.xlabel('f[Hz]')
plt.ylabel('A')

plt.subplot(1,3,2)
y = t*0;
plt.title(r'$A=[1,\frac{1}{2},\frac{1}{3},\frac{1}{4},\frac{1}{5}...]$')
for i in range(1, 1000):
    y += np.sin(2*np.pi*t*f*i)/i
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(0,xlimmax)
plt.xlabel('f[Hz]')

plt.subplot(1,3,3)
y = t*0;
plt.title(r'$A=[1,0,\frac{1}{3},0,\frac{1}{5},0...]$')
for i in range(1, 1000, 2):
    y += np.sin(2*np.pi*t*f*i)/i
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(0,xlimmax)
plt.xlabel('f[Hz]')

plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/fft.png')
CLOSE_PLOTS()













# NOHARMONIC WAVES2
plt.figure(figsize=(7,2))

plt.subplot(1,3,1)
y, sr, t = loadSample('src_extra/Uncategorized 010.flac')
plt.title('Miauczenie kota')
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(50,2000)
plt.xticks([50,1000,2000])
plt.xlabel('f[Hz]')
plt.ylabel('A')

plt.subplot(1,3,2)
y, sr, t = loadSample('src_extra/Uncategorized 021.flac')
plt.title('Pluśnięcie wodą')
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(50,2000)
plt.ylim(0,0.006)
plt.xticks([50,1000,2000])
plt.xlabel('f[Hz]')

plt.subplot(1,3,3)
y, sr, t = loadSample('src/Tom/Tom 0000.flac')
plt.title('Kocioł')
(fs, a) = FFT(t, y)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(fs,a)
plt.xlim(50,2000)
plt.xticks([50,1000,2000])
plt.xlabel('f[Hz]')

plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/fft_nohamronic.png')
CLOSE_PLOTS()












# GŁOSCNOSC
plt.figure(figsize=(7,2))
y = np.arange(1, 141, 1)
t = 10**-12*10**(y/10)
plt.plot(y,t)
plt.xticks([0,25,55,85,140])
plt.ylabel(r'I [$\frac{W}{m^2}$]')
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[0] = '0dB'
labels[1] = '25dB\nSzept'
labels[2] = '55dB\nLodówka'
labels[3] = '85dB\nRuch uliczny'
labels[4] = '140dB\nSilnik odrzutowy z 30m'
ax.set_xticklabels(labels)
plt.xlabel('L [dB]')
plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/loudness.png')
CLOSE_PLOTS()












# PIANO KEYS
plt.figure(figsize=(7,1.5))
t = np.arange(0, 88, 1)
y = 27.5*2**(t/12)
t = t - 48
plt.scatter(y,t, marker='.')
plt.ylabel('k')
plt.xlabel(r'$f_{Bk}$ [Hz]')
plt.yticks([-48, 0, 39])
plt.scatter(440, 0, marker='o', color='black')
plt.text(270, 15, '440 Hz')
plt.hlines(0,0,440, color='black', alpha=0.4, linestyles='dashed')
plt.xlim(0,4300)
plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/piano_freqs.png')
CLOSE_PLOTS()












# SAMPLE RATE
plt.figure(figsize=(7,6))
t = np.arange(0, 1.01, 0.01)
y = np.sin(t * np.pi*2 * 10)
tmax = 1

w = t*0+1
y2=y*w
plt.subplot(3,3,1)
plt.title('Okno prostokątne')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='lightgray', label=r'$y_D(n)$')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = 0.5*(1-np.cos(2*np.pi*t/tmax))
y2=y*w
plt.subplot(3,3,2)
plt.title('Okno Hanninga')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.hamming(len(y))
y2=y*w
plt.subplot(3,3,3)
plt.title('Okno Hamminga')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.bartlett(len(y))
y2=y*w
plt.subplot(3,3,4)
plt.title('Okno Bartletta')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.blackman(len(y))
y2=y*w
plt.subplot(3,3,5)
plt.title('Okno Blackmana')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = signal.flattop(len(y))
y2=y*w
plt.subplot(3,3,6)
plt.title('Okno Flat-top')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.kaiser(len(y), 20)
y2=y*w
plt.subplot(3,3,7)
plt.title(r'Okno Kaisera $\beta=20$')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.kaiser(len(y), 40)
y2=y*w
plt.subplot(3,3,8)
plt.title(r'Okno Kaisera $\beta=40$')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

w = np.kaiser(len(y), 60)
y2=y*w
plt.subplot(3,3,9)
plt.title(r'Okno Kaisera $\beta=60$')
plt.hlines(0, 0, 100, alpha=.2)
plt.plot(y, color='black', alpha=.1)
plt.plot(y, color='lightgray')
plt.plot(y2)
plt.plot(w)
plt.xlim(0,100)
plt.xlabel('n')

plt.tight_layout()
plt.savefig('plots/windows.png')
CLOSE_PLOTS()













# DFT
def FFT2(t, y):
    n = len(t)
    dt = t[1] - t[0]
    
    f = np.fft.fftfreq(n, dt)
    
    a = np.fft.fft(y)
    return (f, a)

plt.figure(figsize=(7,4))
t1 = np.arange(0, 1.1, 0.001)
t2 = np.linspace(0, 1, 9, endpoint=False)
y1 = np.cos(2*np.pi*t1*1 - np.pi/2) + 0.5*np.sin(4*np.pi*t1*1) 
y2 = np.cos(2*np.pi*t2*1 - np.pi/2) + 0.5*np.sin(4*np.pi*t2*1) 
f, a = FFT2(t2,y2)
dt = t2[1] - t2[0]

plt.subplot(2,2,1)
plt.title('L=9, SR=9Hz\n$y(t)=1\cos(2\pi t - \pi/2) + 0.5 \sin(4\pi t)$', fontsize=10)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t1,y1, color='lightgray', zorder=1)
plt.scatter(t2,y2, zorder=2)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')


r = np.real(a)
i = np.imag(a)
_xlim = np.max(np.abs(r))*1.1
_ylim = np.max(np.abs(i))*1.3
plt.subplot(2,2,2)
plt.title('DFT')
plt.hlines(0, -_xlim, _xlim, alpha=.2)
plt.vlines(0, -_ylim, _ylim, alpha=.2)
for i, _f in enumerate(f):
    _f = int(_f)
    s = 300
    color = 'tab:blue'
    if(_f < 0):
        s = 600

    plt.scatter(np.real(a[i]), np.imag(a[i]), color=color, marker=f'${_f}Hz$', s=s)
plt.ylabel('Imag{DFT}')
plt.xlabel('Real{DFT}')
plt.xlim(-_xlim, _xlim)
plt.ylim(-_ylim, _ylim)

plt.subplot(2,2,3)
plt.title(r'$ \frac{1}{SR} $|DFT|')
plt.hlines(0, -4, 4, alpha=.2)
plt.stem(f, np.abs(a)*dt,  basefmt=" ")
plt.ylabel('A')
plt.xlabel('f [Hz]')
plt.xticks([-4,-2,0,1,2,3,4])
plt.yticks([0, 0.25, 0.5])

plt.subplot(2,2,4)
plt.title('Kąt{DFT}')
plt.hlines(0, -4, 4, alpha=.2)
plt.stem(f, np.angle(a),  basefmt=" ")
plt.ylabel('Faza [rad]')
plt.xlabel('f [Hz]')
plt.yticks([-3.14/2, 0, 3.14/2], ['-π/2', '0', 'π/2'])
plt.xticks([-4,-2,0,1,2,3,4])

plt.tight_layout()
plt.savefig('plots/dft.png')
CLOSE_PLOTS()













# DFT Rozdzielczoć
plt.figure(figsize=(7,2))
s = np.array([256,512,1024,2048,4096,8192])
detf = 1/(1/44100 * s)
plt.plot(s,detf, label='Δf [Hz]', marker='o')
plt.plot(s,s/44100*1000, label='Δt [ms]', marker='o')
plt.xticks([256,1024,2048,4096,8192])
plt.xlabel('N')
plt.legend()
plt.tight_layout()
plt.savefig('plots/temp_freq_res_graph.png')
CLOSE_PLOTS()












# DFT
plt.figure(figsize=(7,5))
f = 15

plt.subplot(3,2,1)

tmax=int(1)
t1 = np.linspace(0, tmax, 1000, endpoint=False)
t2 = np.arange(0, tmax, 0.02)
y1 = np.cos(2*np.pi*t1*f - np.pi/2)
y2 = np.cos(2*np.pi*t2*f - np.pi/2)
plt.title(f'L={len(y2)}, SR=50Hz, tmax=1s\n15Hz → $y(t)=\sin(30\pi t)$', fontsize=10)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t1,y1, color='lightgray', zorder=1)
plt.scatter(t2,y2, zorder=2, marker='.')
plt.xlim(0,tmax)
plt.xlabel('t[s]')
plt.ylabel('y')

fs, a = FFT(t2,y2)
plt.subplot(3,2,2)
plt.title(r'$A_{DFT}$')
plt.hlines(0, -4, 4, alpha=.2)
plt.stem(fs, a*2,  basefmt=" ", markerfmt='.')
plt.ylabel('A')
plt.xlabel('f [Hz]')
plt.xticks([1,15.00,25])

plt.subplot(3,2,3)
tmax=0.95
t1 = np.linspace(0, tmax, 1000, endpoint=False)
t2 = np.arange(0, tmax, 0.02)
y1 = np.cos(2*np.pi*t1*f - np.pi/2)
y2 = np.cos(2*np.pi*t2*f - np.pi/2)
plt.title(f'L={len(y2)}, SR=50Hz, tmax=0.95s\n15Hz → $y(t)=\sin(30\pi t)$', fontsize=10)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t1,y1, color='lightgray', zorder=1)
plt.scatter(t2,y2, zorder=2, marker='.')
plt.xlim(0,tmax)
plt.xlabel('t[s]')
plt.ylabel('y')

fs, a = FFT(t2,y2)
plt.subplot(3,2,4)
plt.title(r'$A_{DFT}$')
plt.hlines(0, -4, 4, alpha=.2)
plt.stem(fs, a*2,  basefmt=" ", markerfmt='.')
plt.ylabel('A')
plt.xlabel('f [Hz]')
plt.xticks([1.04167,14.5833,25])

plt.subplot(3,2,5)
tmax=0.95
t1 = np.linspace(0, tmax, 1000, endpoint=False)
t2 = np.arange(0, tmax, 0.02)
y1 = np.cos(2*np.pi*t1*f - np.pi/2)
y2 = np.cos(2*np.pi*t2*f - np.pi/2)
W = np.hanning(len(y1))
y1 = np.hanning(len(y1))*y1
y2 = np.hanning(len(y2))*y2
plt.title(f'L={len(y2)}, SR=50Hz, tmax=0.95s\n15Hz → $y(t)=WF(t)\sin(30\pi t)$', fontsize=10)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t1,y1, color='lightgray', zorder=1)
plt.plot(t1,W, zorder=2, color='tab:orange')
plt.scatter(t2,y2, zorder=3, marker='.')
plt.xlim(0,tmax)
plt.xlabel('t[s]')
plt.ylabel('y')

fs, a = FFT(t2,y2)
plt.subplot(3,2,6)
plt.title(r'$A_{DFT}$')
plt.hlines(0, -4, 4, alpha=.2)
plt.stem(fs, a*2,  basefmt=" ", markerfmt='.')
plt.ylabel('A')
plt.xlabel('f [Hz]')
plt.xticks([1.04167,14.5833,25])

plt.tight_layout()
plt.savefig('plots/dft_windows_sync_async.png')
CLOSE_PLOTS()














# CORRELATION
plt.figure(figsize=(7,4))
t = np.arange(-np.pi*2, np.pi*2, 0.01)
y = 3*np.pi*np.cos(t)

plt.subplot(3,1,1)
plt.title(r"$\int_0^{6\pi} \sin (x) \sin (x+a) \mathrm{d}x = 3 \pi \cos (a)$")
plt.plot(t,y)
plt.xticks(
    [-4/2*np.pi, -3/2*np.pi, -2/2*np.pi, -1/2*np.pi, 0, 1/2*np.pi, 2/2*np.pi, 3/2*np.pi, 4/2*np.pi],
    ['-2π', '-3/2π', '-π', '-1/2π', '0', '1/2π', 'π', '3/2π', '2π'])
plt.xlim(-4/2*np.pi, 4/2*np.pi)
plt.xlabel('a')
plt.ylabel('Korelacja')
plt.hlines(0, -4/2*np.pi, 4/2*np.pi, alpha=.2)
plt.hlines(0, -4/2*np.pi, 4/2*np.pi, alpha=.2)
plt.vlines(0, -10, 10, alpha=.2)
plt.vlines(1/2*np.pi, -10, 10, alpha=.2)
plt.vlines(np.pi, -10, 10, alpha=.2)
plt.ylim(-10,10)

t = np.arange(0, np.pi*6, 0.01)
plt.subplot(3,3,4)
plt.title(r'$a=0$')
y1 = np.sin(t)
y2 = np.sin(t)
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1, label='sin(x)')
plt.plot(t,y2, label='sin(x+a)')
#plt.plot(t,y1*y2)
plt.xlabel('x')

y2 = np.sin(t + 1/2*np.pi)
plt.subplot(3,3,5)
plt.title(r'$a=\frac{1}{2}\pi$')
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1, label='sin(x)')
plt.plot(t,y2, label='sin(x+a)')
#plt.plot(t,y1*y2)
plt.xlabel('x')

y2 = np.sin(t + np.pi)
plt.subplot(3,3,6)
plt.title(r'$a=\pi$')
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1, label='sin(x)')
plt.plot(t,y2, label='sin(x+a)')
#plt.plot(t,y1*y2)
plt.xlabel('x')

plt.subplot(3,3,7)
plt.title(r'$a=0$')
y1 = np.sin(t)
y2 = np.sin(t)
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1*y2, color='tab:green')
plt.xlabel('x')
plt.ylim(-1.2,1.2)
plt.gca().fill_between(t, 0, y1*y2, color='tab:green', alpha=0.3)

y2 = np.sin(t + 1/2*np.pi)
plt.subplot(3,3,8)
plt.title(r'$a=\frac{1}{2}\pi$')
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1*y2, color='tab:green')
plt.xlabel('x')
plt.ylim(-1.2,1.2)
plt.gca().fill_between(t, 0, y1*y2, color='tab:green', alpha=0.3)

y2 = np.sin(t + np.pi)
plt.subplot(3,3,9)
plt.title(r'$a=\pi$')
plt.hlines(0, 0, np.pi*6, alpha=.2)
plt.plot(t,y1*y2, color='tab:green')
plt.xlabel('x')
plt.ylim(-1.2,1.2)
plt.gca().fill_between(t, 0, y1*y2, color='tab:green', alpha=0.3)

plt.tight_layout()
plt.savefig('plots/corelaton_example.png')
CLOSE_PLOTS()












# CORRELATION
plt.figure(figsize=(8,4))
y, sr, t = loadSample('src/Snare/Snare 0176.flac')

plt.subplot(2,2,1)
plt.hlines(0, 0, np.max(t), alpha=.2)
plt.plot(t,y)
plt.xlim(0,0.2)
plt.xlabel('t [s]')
plt.ylabel('y')
plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
plt.title('Spróbkowany sygnał')

S = librosa.stft(y)/sr*2
A = np.abs(S)
P = np.angle(S)

plt.subplot(2,2,2)
librosa.display.specshow(P, sr=sr, y_axis='log', x_axis='time')
plt.xlim(0,0.2)
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f rad')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(r'$φ_{STFT}$')
plt.yticks([0, 128, 512, 2048, 8192])

plt.subplot(2,2,3)
librosa.display.specshow(S, sr=sr, y_axis='log', x_axis='time')
plt.xlim(0,0.2)
plt.title('Power spectrogram')
plt.colorbar(format='%2.3f')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(r'$A_{STFT}$')
plt.yticks([0, 128, 512, 2048, 8192])

plt.subplot(2,2,4)
librosa.display.specshow(librosa.amplitude_to_db(S), sr=sr, y_axis='log', x_axis='time')
plt.xlim(0,0.2)
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f db')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(r'$GWM_{log}(A_{STFT})$')
plt.yticks([0, 128, 512, 2048, 8192])

plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/stft_example.png')
CLOSE_PLOTS()












# PIANO KEYS
t = np.arange(0, 10, 1)
y = 31.25*2**t

plt.figure(figsize=(7,2.5))
plt.subplot(2,1,1)
plt.stem(y, y*0+1,  basefmt=" ")
plt.xlabel(r'f [Hz]')
plt.ylim(0, 1.3)

y = 2595*np.log10(1+y/700)
plt.subplot(2,1,2)
plt.stem(y, y*0+1, basefmt=" ")
plt.xlabel(r'm [mel]')
plt.ylim(0, 1.3)

# plt.scatter(c, 0, marker='o', color='black')
# plt.text(270, 15, '440 Hz')
# plt.hlines(0,0,c, color='black', alpha=0.4, linestyles='dashed')
# plt.xlim(0,4300)
plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/piano_freqs_mel.png')
CLOSE_PLOTS()












# PIANO KEYS
x = np.arange(0, 20000, 1)
y = 2595*np.log10(1+x/700)

plt.figure(figsize=(7,2))
plt.plot(x, y)
plt.xlabel(r'f [Hz]')
plt.ylabel(r'm [mel]')
plt.xlim(0, 20000)
plt.yticks([0, 1000,2000,3000,4000])
plt.xticks([0, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
plt.hlines(1000, 0, 1000, alpha=.2)
plt.vlines(1000, 0, 1000, alpha=.2)
plt.ylim(0,4000)
plt.tight_layout()
plt.savefig('plots/hz_mel.png')
CLOSE_PLOTS()












# MEL FB
sr = 44100
wl = 2048 
f = librosa.fft_frequencies(sr,wl)

melfb = librosa.filters.mel(44100, wl, n_mels=7)

plt.figure(figsize=(7,2))

for melf in melfb:
    plt.plot(f, melf)
plt.xlabel('f [Hz]')
plt.ylabel('Waga')

plt.tight_layout()
plt.savefig('plots/mel_filters.png')
CLOSE_PLOTS()













# CORRELATION
plt.figure(figsize=(8,3))
y, sr, t = loadSample('src/Snare/Snare 0088.flac')
changeLength(y, int(sr*0.25))

hop_length=128

n_fft=2048
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)/sr*2
S = np.abs(S)**2

mels = 128

plt.subplot(1,3,2)
librosa.display.specshow(librosa.power_to_db(S), sr=sr, hop_length=hop_length, y_axis='linear', x_axis='time', fmax=22500)
# plt.xlim(0,0.2)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('$GWM_{log}(A_{STFT})$\n[1025 x 82]')

plt.subplot(1,3,3)
S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=mels)
librosa.display.specshow(librosa.power_to_db(S), sr=sr, hop_length=hop_length, y_axis='mel', x_axis='time', fmax=22500)
# plt.xlim(0,0.2)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('$GWM_{log}(A_{Mel})$\n[128 x 82]')

plt.tight_layout()
plt.savefig('plots/spect_to_melspec.png')
CLOSE_PLOTS()













# CORRELATION
plt.figure(figsize=(8,3))
y, sr, t = loadSample('src/Snare/Snare 0088.flac')
changeLength(y, int(sr*0.25))

hop_length=128
mels = 128
n_fft=2048
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)/sr*2
S = np.abs(S)**2
S = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=mels))

plt.subplot(1,2,1)
librosa.display.specshow(S, sr=sr, hop_length=hop_length, y_axis='mel', x_axis='time', fmax=22500)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('$GWM_{log}(A_{Mel})$\n[128 x 82]')
plt.colorbar(format='%+2.0f db')

S2 = librosa.feature.mfcc(S=S, sr=sr, n_mfcc=20, norm=None, dct_type=2)

plt.subplot(1,2,2)
librosa.display.specshow(S2, sr=sr, hop_length=hop_length, x_axis='time')
plt.xlabel('t [s]')
plt.title('$MFCC$\n[20 x 82]')
plt.ylabel('Wiersz')
plt.yticks(np.arange(1,21,2))
plt.colorbar()

plt.tight_layout()
plt.savefig('plots/mel_mfcc.png')
CLOSE_PLOTS()










# CORRELATION
plt.figure(figsize=(8,3))
y, sr, t = loadSample('src/Snare/Snare 0088.flac')
changeLength(y, int(sr*0.25))

S = librosa.cqt(y, sr=sr, hop_length=512, fmin=None, n_bins=110, bins_per_octave=12)
S = librosa.amplitude_to_db(np.abs(S), ref=np.max) 

plt.subplot(1,2,1)
librosa.display.specshow(S, sr=sr, hop_length=hop_length, y_axis='mel', x_axis='time', fmax=22500)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('$GWM_{log}(A_{CQT})$\n[128 x 82]')
plt.colorbar(format='%+2.0f db')

S2 = librosa.feature.mfcc(S=S, sr=sr, n_mfcc=20, norm=None, dct_type=2)

plt.subplot(1,2,2)
librosa.display.specshow(S2, sr=sr, hop_length=hop_length, x_axis='time')
plt.xlabel('t [s]')
plt.title('$CQCC$\n[20 x 82]')
plt.ylabel('Wiersz')
plt.yticks(np.arange(1,21,2))
plt.colorbar()

plt.tight_layout()
plt.savefig('plots/cqt_cqcc.png')
CLOSE_PLOTS()











file = 'src/Kick/Kick 0001.flac'
audio, sr, _ = loadSample(file)
audio = changeLength(audio)
hop_length = 512
    
# Wizualizacja precyzji częstotliwosciowej
def stftFrequencyPrecission(a=20,b=356):

    plt.figure(figsize=(7,2))
    for i, n_fft in enumerate([8192, 4096, 2048, 2048//2, 2048//4]):
        plt.subplot(1,5,i+1)
        # Obliczenie długoci okna
        win_length = np.round(n_fft / config.sample_rate * 1000, decimals=2) 
        # Transformata
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
        plt.ylim(a,b)
        plt.xlim(0, 0.3)
        plt.xticks([0,0.1,0.2,0.3])
        plt.title(f'STFT {n_fft}\n Δt={win_length}ms', size=9.5)
        plt.ylabel('f [Hz]')
        if(i > 0):
            plt.yticks([])
            plt.ylabel(None)
        plt.xlabel('t [s]')
    plt.tight_layout()  

stftFrequencyPrecission()
plt.savefig('plots/STFT_FrequencyPrecision_HL512.png')
CLOSE_PLOTS()

# Wizualizacja precyzji częstotliwosciowej
def stftTemporalPrecission():
    plt.figure(figsize=(7,2))
    for i, n_fft in enumerate([8192, 4096, 2048, 2048//2, 2048//4]):
        plt.subplot(1,5,i+1)
        # Obliczenie długoci okna
        win_length = np.round(n_fft / config.sample_rate * 1000, decimals=2) 
        # Transformata
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', hop_length=hop_length)
        plt.xlim(0, 0.1)
        plt.xticks([0, 0.05, 0.1])
        plt.yticks([0, 128, 512, 2048, 8192])
        plt.title(f'STFT {n_fft}\n Δt={win_length}ms', size=9.5)
        plt.ylabel('f [Hz]')
        if(i > 0):
            plt.yticks([])
            plt.ylabel(None)
        plt.xlabel('t [s]')
    plt.tight_layout()  

stftTemporalPrecission()
plt.savefig('plots/STFT_TemporalPrecision_HL512.png')
CLOSE_PLOTS()

# Wizualizacja precyzji częstotliwosciowej
def stftTemporalPrecission2():
    plt.figure(figsize=(7,2))
    for i, n_fft in enumerate([8192, 4096, 2048, 2048//2, 2048//4]):
        plt.subplot(1,5,i+1)
        # Obliczenie długoci okna
        win_length = np.round(n_fft / config.sample_rate * 1000, decimals=2) 
        # Transformata
        S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft//2)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', hop_length=n_fft//2)
        plt.xlim(0, 0.1)
        plt.xticks([0, 0.05, 0.1])
        plt.yticks([0, 128, 512, 2048, 8192])
        plt.title(f'STFT {n_fft}\n Δt={win_length}ms', size=9.5)
        plt.ylabel('f [Hz]')
        if(i > 0):
            plt.yticks([])
            plt.ylabel(None)
        plt.xlabel('t [s]')
    plt.tight_layout()  

stftTemporalPrecission2()
plt.savefig('plots/STFT_TemporalPrecision_HLx2.png')
CLOSE_PLOTS()

file = 'src/Clap/Clap 0001.flac'
audio, sr, _ = loadSample(file)
audio = changeLength(audio)
hop_length = 512

stftFrequencyPrecission(4096,16384)
plt.savefig('plots/STFT_FrequencyPrecision_High_HL512.png')
CLOSE_PLOTS()












# CORRELATION
plt.figure(figsize=(6,4))

plt.subplot(2,3,1)
K = np.ones((25,25))/25/25
y,x = K.shape
y = int((y+1)/2)
x = int((x+1)/2)
plt.title(f'Filtr [{K.shape[0]}, {K.shape[1]}]')
plt.plot(0,0)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
A = image.imread('src_extra/test_image.jpg')
A = np.copy(A).astype(float)/255
plt.title(f'Wejście [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
for i in range(3):
    A[:,:,i] = scipy.ndimage.filters.convolve(A[:,:,i], K)
A = A[y:-y,x:-x,:]
plt.title(f'Wynik [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A)
plt.xticks([])
plt.yticks([])


plt.subplot(2,3,4)
K = np.array([[0,4,0],[4,-16,4],[0,4,0]])
y,x = K.shape
y = int((y+1)/2)
x = int((x+1)/2)
plt.title(f'Filtr [{K.shape[0]}, {K.shape[1]}]')
plt.plot(0,0)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,5)
A = image.imread('src_extra/test_image.jpg')
A = np.copy(A).astype(float)/255
plt.title(f'Wejście [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,6)
for i in range(3):
    A[:,:,i] = scipy.ndimage.filters.convolve(A[:,:,i], K)
A = A[y:-y,x:-x,:]
plt.title(f'Wynik [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A)
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig('plots/example_convolutons.png')
CLOSE_PLOTS()













# CORRELATION
plt.figure(figsize=(8,3))

K = np.array([[2,-2,0],[2,-2,0],[2,-2,0]])
y,x = K.shape
y = int((y+1)/2)
x = int((x+1)/2)
A = image.imread('src_extra/test_image2.jpeg')
A = np.copy(A).astype(float)/255
A = np.mean(A,axis=2)

plt.subplot(1,3,1)
plt.title(f'Przed splotem [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A, cmap='binary_r')
plt.xticks([])
plt.yticks([])
A = A[1:-1,1:-1]

A = scipy.ndimage.filters.convolve(A, K)
A[A<0] = 0

plt.subplot(1,3,2)
plt.title(f'Wejście [{A.shape[0]}, {A.shape[1]}]')
plt.imshow(A, cmap='binary_r')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
B = skimage.measure.block_reduce(A, (4,4), np.max)
B = B[0:-1,0:-1]
plt.title(f'Max-pooling 4x4 [{B.shape[0]}, {B.shape[1]}]')
plt.imshow(B, cmap='binary_r')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig('plots/max_pool_example.png')
CLOSE_PLOTS()













# CORRELATION
plt.figure(figsize=(8,3))
hop_length=1024
mels = 128
n_fft=2048

plt.subplot(1,2,1)
y, sr, t = loadSample('src/Snare/Snare 0088.flac')
y = changeLength(y, int(sr*0.25))
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)/sr*2
S = np.abs(S)**2
S = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=mels))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, y_axis='mel', x_axis='time', fmax=22500)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('MEL-S [128 x 11]')

plt.subplot(1,2,2)
y, sr, t = loadSample('src/Kick/Kick 0588.flac')
y = changeLength(y, int(sr*0.25))
S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)/sr*2
S = np.abs(S)**2
S = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr, n_mels=mels))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, y_axis='mel', x_axis='time', fmax=22500)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('MEL-S [128 x 11]')

plt.tight_layout()
plt.savefig('plots/kick_snare_mel-s.png')
CLOSE_PLOTS()












# CORRELATION
plt.figure(figsize=(8,4))
x = np.arange(-3,3,0.01)

plt.subplot(2,3,1)
y=x>0
plt.plot(x,y)
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.title('Skok jednostkowy\n$y \in \{0,1\}$')
plt.xlim(-1,1)
plt.ylim(-0.1,1.1)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.subplot(2,3,2)
y=1.0/(1.0 + np.exp(-x))
plt.plot(x,y)
plt.title('Funkcja sigmoidalna\n$y \in (0,1)$')
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.ylim(-0.1,1.1)
plt.xlim(-3,3)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.subplot(2,3,3)
y=x*(x>0)
plt.plot(x,y)
plt.title('ReLU\n$y \in [0,Inf)$')
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.xlim(-1,1)
plt.ylim(-0.1,1)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.subplot(2,3,4)
y=x*(x>0) + 0.1*x*(x<0)
plt.plot(x,y)
plt.title('Leaky ReLU\n$y \in (-Inf,Inf)$')
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.xlim(-1,1)
plt.ylim(-0.15,1)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.subplot(2,3,5)
y=(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
plt.plot(x,y)
plt.title('Tanh\n$y \in (-1,1)$')
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.ylim(-1.1,1.1)
plt.xlim(-1.5,1.5)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.subplot(2,3,6)
y=np.arctan(x)
plt.plot(x,y)
plt.title('Arctg\n$y \in (-1,1)$')
plt.hlines(0, np.min(x), np.max(x), alpha=.2)
plt.vlines(0, -2, 2, alpha=.2)
plt.ylim(-1.1,1.1)
plt.xlim(-1,1)
plt.xlabel('x')
plt.ylabel('AF(x)')

plt.tight_layout()
plt.savefig('plots/activation_func.png')
CLOSE_PLOTS()












# MIN FINDING
plt.figure(figsize=(8,3))
def fx(x):
    return (x-2)*(x+2)
def tangent(p):
    a = 2*p
    x = np.arange(p-0.75,p+0.75,0.01)
    y = a*(x-p) + fx(p)
    plt.plot(x,y, color='tab:orange', zorder=2)

x = np.arange(-4,4,0.01)
y = fx(x)
plt.plot(x,y, zorder=1)

tangent(-3)
tangent(0)
tangent(3)
x = np.array([-3,0,3])
y = fx(x)
plt.scatter(x,y, marker='o', s=80, zorder=3)

plt.text(-3, 5, '  $g\'(-3)=-6$', fontsize=12, ha='left', va='bottom')
plt.text(3, 5, '$g\'(3)=6$  ', fontsize=12, ha='right', va='bottom')
plt.text(0, -4+1, '$g\'(0)=0$', fontsize=12, ha='center', va='bottom')

plt.title('$Funkcja: g(x)=x^2-4$\nPochodna: $g\'(x)=\\frac{dg(x)}{dx}=2x$')
plt.xlabel('x')
plt.ylabel('g(x)')
plt.xlim(-4,4)
plt.ylim(-6,12)
plt.tight_layout()
plt.savefig('plots/min_find.png')
CLOSE_PLOTS()












# MANY MINS
plt.figure(figsize=(8,2))
x = np.arange(-6,6,0.01)
y = fx(x) + 3*np.sin(np.pi*x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.ylim(-12,31)
plt.xlim(-6,6)
plt.tight_layout()
plt.savefig('plots/many_mins.png')
CLOSE_PLOTS()












#MAE log loss
plt.figure(figsize=(8,2))
x = np.arange(0,1,0.01)

plt.subplot(1,2,1)
y = np.abs(1 - x)
plt.plot(x,y,label='$|1-x|$')
y = np.abs(0 - x)
plt.plot(x,y,label='$|0-x|$')
plt.legend()
plt.title('Średni błąd bezwzględny')
plt.xlabel('x')
plt.xlim(0,1)

plt.subplot(1,2,2)
y = -np.log(np.abs(1 + x - 1))
plt.plot(x,y,label='$-ln|1+x-1|$')
y = -np.log(np.abs(0 + x - 1))
plt.plot(x,y,label='$-ln|0+x-1|$')
plt.legend()
plt.title('Strata logarytmiczna')
plt.xlabel('x')
plt.xlim(0,1)

plt.tight_layout()
plt.savefig('plots/mae_logloss.png')
CLOSE_PLOTS()












# EMA
def ema(x,beta):
    v = np.zeros(len(x))
    v[0] = x[0]
    for i in range(1,len(v)):
        v[i] = beta*v[i-1] + (1-beta)*x[i]
    return v
def fx(x):
    return (x-2)*(x+2)*(x+2)*(x-2)
plt.figure(figsize=(8,2.5))

plt.subplot(1,3,1)
x = np.arange(-4,4,0.05)
y = fx(x)
x = np.array(range(len(x)))
np.random.seed(43)
y2 = y + np.random.normal(0,10,len(y))
plt.scatter(x,y2,marker='.', label='$ v_i $')
plt.plot(x,y, color='tab:red',linewidth=3, label='Sygnał')
plt.legend()
plt.xlabel('i')
plt.title('Sygnał z szumem Gaussa')

plt.subplot(1,3,2)
v = ema(y2,0.5)
plt.scatter(x,v, marker='.', color=getTabColor(2), label='$c=0.5$')
v = ema(y2,0.9)
plt.scatter(x,v, marker='.', color=getTabColor(1), label='$c=0.9$')
v = ema(y2,0.975)
plt.scatter(x,v, marker='.', color=getTabColor(4), label='$c=0.975$')
plt.legend()
plt.xlabel('i')
plt.title('$ \\tilde v_i $')


plt.subplot(1,3,3)
v = ema(y2,0.9)
plt.scatter(x,v, marker='.', color=getTabColor(1), label='$ \\tilde v_i $ $(c=0.9)$')
plt.plot(x,y, color='tab:red',linewidth=2, label='Sygnał')
plt.title('Porównanie sygnału z $ \\tilde v_i $')
plt.legend()

plt.tight_layout()
plt.savefig('plots/moving_exp_avrg.png')
CLOSE_PLOTS()













#LINREG
def getErros(y, y_pred):
    err  = y - y_pred
    errb = np.zeros([2,len(err)])
    errb[1,:] = np.clip(err, 0, None)
    errb[0,:] = -np.clip(err, None, 0)
    return (np.mean(np.abs(err)), errb)
x1 = np.arange(0, 20, 2)
x2 = np.arange(1, 20, 2)
x3 = np.arange(0, 20, 1)
np.random.seed(42)
y1 = x1 + np.random.normal(0,15,len(x1))
np.random.seed(43)
y2 = x2 + np.random.normal(0,15,len(x2))
min_val = np.min(np.concatenate([y1,y2]))
y1 = y1 - min_val
y2 = y2 - min_val

plt.figure(figsize=(7,4))
plt.subplot(1,3,1)
plt.scatter(x1,y1, label='Zbiór uczący', color='tab:blue', s=15)
plt.scatter(x2,y2, label='Zbiór testowy', color='tab:orange', s=15)
plt.title('Dane')
plt.legend()
plt.ylim(-2,60)

regr = LinearRegression()
regr.fit(x1.reshape(-1, 1), y1.reshape(-1, 1))
y1_pred = regr.predict(x1.reshape(-1, 1))[:,0]
y2_pred = regr.predict(x2.reshape(-1, 1))[:,0]

plt.subplot(2,3,2)
err, error_bars = getErros(y1, y1_pred)
plt.errorbar(x1,y1_pred, error_bars, label='LinReg', color='tab:green', zorder=1)
plt.scatter(x1,y1, color='tab:blue', s=15, zorder=2)
plt.title(f'Regresja liniowa (Uczenie)\n$MAE_{{Train}}={np.round(err,2)}$')
plt.ylim(-2,60)

plt.subplot(2,3,3)
err, error_bars = getErros(y2, y2_pred)
plt.errorbar(x2,y2_pred, error_bars, label='LinReg', color='tab:green', zorder=1)
plt.scatter(x2,y2, color='tab:orange', s=15, zorder=2)
plt.title(f'Regresja liniowa (Test)\n$MAE_{{Test}}={np.round(err,2)}$')
plt.ylim(-2,60)

interp = scipy.interpolate.interp1d(x1, y1, bounds_error=False, fill_value="extrapolate")
y1_pred = interp(x1)
y2_pred = interp(x2)
y3_pred = interp(x3)

plt.subplot(2,3,5)
err, error_bars = getErros(y1, y1_pred)
plt.plot(x3,y3_pred, color='tab:green', zorder=1)
plt.scatter(x1,y1, color='tab:blue', s=15, zorder=2)
plt.title(f'Interpolacja (Uczenie)\n$MAE_{{Train}}={np.round(err,2)}$')
plt.ylim(-2,60)

plt.subplot(2,3,6)
err, error_bars = getErros(y2, y2_pred)
plt.errorbar(x2,y2_pred, error_bars, label='LinReg', color='white', ecolor='tab:green', zorder=0)
plt.plot(x3,y3_pred, color='tab:green', zorder=1)
plt.scatter(x2,y2, color='tab:orange', s=15, zorder=2)
plt.title(f'Interpolacja (Test)\n$MAE_{{Test}}={np.round(err,2)}$')
plt.ylim(-2,60)

plt.tight_layout()
plt.savefig('plots/overfit.png')
CLOSE_PLOTS()













# CORRELATION
plt.figure(figsize=(6,3))
src, sr, t = loadSample('src/Snare/Snare 0288.flac')


y = changeLength(src, int(sr*0.3))
t = np.arange(0,len(y))/sr

plt.subplot(2,2,1)
plt.plot(t,y)
plt.xticks([])
plt.ylim(-1.1,1.1)
plt.ylabel('y')
plt.xlim(0,0.3)
plt.title('Sygnał')

plt.subplot(2,2,3)
CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
librosa.display.specshow(CQT, x_axis='time', sr=sr, y_axis='cqt_hz')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('CQT')


y = interpSample(src, 12)
y = changeLength(y, int(sr*0.3))
t = np.arange(0,len(y))/sr

plt.subplot(2,2,2)
plt.plot(t,y)
plt.xticks([])
plt.ylim(-1.1,1.1)
plt.ylabel('y')
plt.xlim(0,0.3)
plt.title('Sygnał - szybkość 200%')

plt.subplot(2,2,4)
CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
librosa.display.specshow(CQT, x_axis='time',  sr=sr, y_axis='cqt_hz')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title('CQT')

plt.tight_layout()
plt.savefig('plots/audio_interpolate.png')
CLOSE_PLOTS()













# DFT
plt.figure(figsize=(7,4))
t1 = np.arange(0, 1.1, 0.001)
t2 = np.linspace(0, 1, 9, endpoint=False)
y1 = np.cos(2*np.pi*t1*1 - np.pi/2) + 0.5*np.sin(4*np.pi*t1*1) 
y2 = np.cos(2*np.pi*t2*1 - np.pi/2) + 0.5*np.sin(4*np.pi*t2*1) 

plt.subplot(1,2,1)
plt.hlines(0, 0, 1, alpha=.2)
plt.plot(t1,y1, color='lightgray', zorder=1)
plt.scatter(t2,y2, zorder=2)
plt.xlim(0,1)
plt.xlabel('t[s]')
plt.ylabel('y')


def DFT(x, dt, perk=100):
    kMax = int(np.floor(len(x)/2)+1)
    kNum = kMax*perk+1
    out = np.zeros(kNum, dtype=complex)
    ks = np.linspace(0,1,kNum)*kMax
    for k, kv in enumerate(ks):
        for n in range(len(x)):
            out[k]= out[k] + x[n] * np.exp(-1j*2.0*np.pi*kv*n/len(x))
            
    fs = ks/dt/len(x)
    return fs, out

plt.subplot(1,2,2)
fs, out = DFT(y2, t2[1]-t2[0])
plt.plot(fs,np.abs(out))
CLOSE_PLOTS()











# CORRELATION
plt.figure(figsize=(9,4))
src, sr, t = loadSample('src/Snare/Snare 0288.flac')

y = changeLength(src, int(sr*0.3))
t = np.arange(0,len(y))/sr

plt.subplot(1,3,1)
S = librosa.stft(y, n_fft=219, hop_length=512, center=False)
S = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S, x_axis='time', sr=sr, y_axis='hz')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(f'STFT\n[{S.shape[0]},{S.shape[1]}]')

plt.subplot(1,3,2)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=110, n_fft=1024, hop_length=512, center=False)
S = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S, x_axis='time', sr=sr, y_axis='mel', fmax=sr/2)
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(f'MEL-S\n[{S.shape[0]},{S.shape[1]}]')

plt.subplot(1,3,3)
S = librosa.cqt(y, sr=sr, n_bins=110, bins_per_octave=12, hop_length=512)
S = librosa.amplitude_to_db(np.abs(S), ref=np.max)
librosa.display.specshow(S, x_axis='time', sr=sr, y_axis='cqt_hz')
plt.xlabel('t [s]')
plt.ylabel('f [Hz]')
plt.title(f'CQT\n[{S.shape[0]},{S.shape[1]}]')

plt.tight_layout()
plt.savefig('plots/stft_mels_cqt.png')
CLOSE_PLOTS()









# CQT MEL CZASY
plt.figure(figsize=(6,2))
times = []
for i in range(100):
   start = timer()
   S = librosa.stft(y, n_fft=219, hop_length=512, center=False)
   S = librosa.amplitude_to_db(np.abs(S), ref=np.max)
   end = timer()
   times.append(end-start)
stft_times=np.array(times)*1000

times = []
for i in range(100):
   start = timer()
   S = librosa.feature.melspectrogram(y, sr=sr, n_mels=110, n_fft=1024, hop_length=512, center=False)
   S = librosa.power_to_db(S, ref=np.max)
   end = timer()
   times.append(end-start)
mels_times=np.array(times)*1000

times = []
for i in range(100):
   start = timer()
   S = librosa.cqt(y, sr=sr, n_bins=110, bins_per_octave=12, hop_length=512)
   S = librosa.amplitude_to_db(np.abs(S), ref=np.max)
   end = timer()
   times.append(end-start)
cqt_times=np.array(times)*1000

plt.boxplot([stft_times, mels_times, cqt_times], whis=[5,95], labels=['STFT', 'MEL-S', 'CQT'])
plt.ylabel('t [ms]')
plt.tight_layout()
plt.savefig('plots/stft_mels_cqt_times.png')
CLOSE_PLOTS()











# WYBRANE REPREZENTACJE
plt.figure(figsize=(6.5,3))
audio, _, _ = loadSample('src/Snare/Snare 0025.flac')
plt.subplot(1,2,1)
S = audioToMELS(audio)
showMELS(S)
plt.colorbar()
plt.subplot(1,2,2)
S = audioToCQT(audio)
showCQT(S)
plt.colorbar()
plt.tight_layout()
plt.savefig('plots/mels_cqt_final.png')
CLOSE_PLOTS()


# WYBRANE REPREZENTACJE
plt.figure(figsize=(8,5))
audio, _, _ = loadSample('src/Snare/Snare 0025.flac')
plt.subplot(2,3,2)
S = audioToMELS(audio)
showMELS(S)
plt.colorbar()
plt.subplot(2,3,1)
S = audioToCQT(audio)
showCQT(S)
plt.colorbar()
plt.subplot(2,3,4)
S = audioToMFCC(audio)
showMFCC(S, 'MFCC')
plt.colorbar()
plt.subplot(2,3,3)
S = audioToCQCC(audio)
showMFCC(S, 'CQCC')
plt.colorbar()
plt.subplot(2,3,5)
S = audioToMFCCL2(audio)
showMFCC(S, 'MFCCL2')
plt.colorbar()
plt.subplot(2,3,6)
S = audioToMFCCMAX(audio)
showMFCC(S, 'MFCCMAX')
plt.colorbar()
plt.tight_layout()
plt.savefig('plots/mels_cqt_mfcc_cqcc_final.png')
plt.subplot(3,1,1)
S = audioToMELS(audio)
showCQT(S)
CLOSE_PLOTS()



plt.figure(figsize=(8,3))
plt.subplot(1,3,1)
S = audioToMFCC(audio)
showMFCC(S)
plt.colorbar()
plt.subplot(1,3,2)
S = audioToMFCCL2(audio)
showMFCC(S, 'MFCCL2')
plt.colorbar()
plt.subplot(1,3,3)
S = audioToMFCCMAX(audio)
showMFCC(S, 'MFCCMAX')
plt.colorbar()
plt.tight_layout()
plt.tight_layout()
plt.savefig('plots/mfcc_final.png')
CLOSE_PLOTS()






# Podgląd plików
randomizer = random.Random(13)

src_path = 'src'
# Lista wszystkich folderÃ³w gÅ‚Ã³wnych typÃ³w dÅºwiÄ™kÃ³w (src/Kick, src/Clap, ...):
mainfolders = list(filter(os.path.isdir, [os.path.join(src_path, p) for p in os.listdir(src_path)]))
# Lista wszystkich podfolderÃ³w w folderach gÅ‚ownych (src/Kick/A, src/Clap/A, ...)
subfolders  = []
for f in mainfolders:
    subfolders.extend(list(filter(os.path.isdir, [os.path.join(f, p) for p in os.listdir(f)])))
    
# Wybranie 10 losowych plikÃ³w z kaÅ¼dej kategorii
files = []
for mf in mainfolders:
    all_files = librosa.util.find_files(mf)
    randomizer.shuffle(all_files)
    files.extend(all_files[0:10])

# Generowanie wykresu
plt.figure(figsize=(8,7))
for i, path in enumerate(files):
    audio, _, _ = loadSample(path)
    audio = changeLength(audio)
    S = audioToCQT(audio)
    plt.subplot(10,10,(i%10)*10 + i//10 + 1)
    showCQT(S,False)
    plt.text(0.8, 0.7, i+1, horizontalalignment='center', weight ='bold',
             verticalalignment='center', transform=plt.gca().transAxes, color='white')
    if(i%10 == 0):
        plt.title(os.path.splitext(os.path.basename(path))[0].split()[0])
    print(i)
plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95, top=0.95, bottom=0.05)    
plt.savefig('plots/Files overview.png')
CLOSE_PLOTS()











mfcc  = np.load('src/mfcc_train_data.npy', allow_pickle=True).item()['times']
mels  = np.load('src/mels_train_data.npy', allow_pickle=True).item()['times']
cqt   = np.load('src/cqt_train_data.npy', allow_pickle=True).item()['times']
load  = np.load('src/load_times.npy', allow_pickle=True)

def plotTimes(data, name):
    print(f'{name}:\t\n\tMedian:\t{np.median(data)}\n\tMean:\t{np.mean(data)}\n')
plotTimes(mfcc,'MFCC')
plotTimes(mels,'MELS')    
plotTimes(cqt,'CQT')    
plotTimes(load,'LOAD')    

