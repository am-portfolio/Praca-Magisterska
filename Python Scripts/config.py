# config.py

import os
import sys
    
################# FOLDER NA WYNIKI ##################

os.makedirs('out', exist_ok=True)
os.makedirs('out_main', exist_ok=True)
os.makedirs('plots', exist_ok=True)

#####################################################
############ ZMIENNE UżYWANE W SKRYPTACH ############
#####################################################

sample_rate = 44100         # Próbkowanie w jakim będą przetwarzane pliki
amplitude_treshold = 0.1    # Próg amplitudy sampli uważanych za szum na początku pliku.      
fadein = 10                 # Czas fadeinu po ucięciu początku [sample]
limit_length = 2000         # Docelowa długość pliku do przetwarzania [milisekundy]
fadeout = 25                # Czas fadeoutu przed zakończeniem pliku [milisekundy]

# Długosc w samplach do jakiej zostaną znormalizowane pliki
length_in_samples = int(sample_rate*limit_length/1000)
fadeout_in_samples = int(sample_rate*fadeout/1000)