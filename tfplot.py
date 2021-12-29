import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spkit
import matplotlib.ticker as ticker
import spkit as sp
from spkit.cwt import ScalogramCWT
from spkit.cwt import compare_cwt_example

file_name = 'blink2.csv'
def cwt_timefreq():
    XW, S = ScalogramCWT(x, t, fs=fs, wType='Gauss', PlotPSD=True, interpolation='sinc')

def plotting(Xf, ch_names):
    plt.figure(figsize=(12, 5))
    plt.plot(t, Xf + np.arange(-3, 3) * 200)
    plt.xlim([t[0], t[-1]])
    plt.xlabel('time (sec)')
    plt.yticks(np.arange(-3, 3) * 200, ch_names)
    plt.grid()
    plt.title('Xf: 14 channel - EEG Signal (filtered)')
    plt.show()

x = []

df = pd.read_csv(file_name)
for i in range(len(df)):
    x.append(df['af8'][i])
x = np.array(x)
fs = 500

t = np.arange(len(x)) / fs


