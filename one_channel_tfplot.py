import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spkit
import matplotlib.ticker as ticker

print('spkit-version ', spkit.__version__)
import spkit as sp
from spkit.cwt import ScalogramCWT
from spkit.cwt import compare_cwt_example

# x, fs = sp.load_data.eegSample_1ch()
x = []
df = pd.read_csv('fp1.csv')
for i in range(len(df)):
    x.append(df['Fp1'][i])
x = np.array(x)
fs = 500

t = np.arange(len(x)) / fs
print(x)
print('shape ', x.shape, t.shape)

# plt.figure(figsize=(15, 3))
# plt.plot(t, x)
# plt.xlabel('time')
# plt.ylabel('amplitude')
# plt.show()

XW,S = ScalogramCWT(x,t,fs=fs,wType='Gauss',PlotPSD=True, interpolation='sinc')

# plt.figure(figsize=(15,3))
# plt.imshow(np.abs(XW),aspect='auto',origin='lower',cmap='jet',interpolation='sinc')
# y_tick = []
# for i in range(10):
#     y_tick.append((i+1)*10)
# plt.yticks(y_tick)
# plt.show()

