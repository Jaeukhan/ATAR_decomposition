import os
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
import pandas as pd
import sys
from scipy.fft import fft, ifft


def psd_mean_save(x, Y, freq, k, i, file):
    df_beta = pd.DataFrame(index=range(1, 6), columns=x.columns)
    beta_y = Y[freq[0] * 10: freq[1] * 10]
    beta_mean = beta_y.mean() * 10 ** 8
    df_beta.iloc[k, i] = beta_mean
    df_beta.to_csv(f'result/psd/{file[:-4]}beta.csv', index=False)
