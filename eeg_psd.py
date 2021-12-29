import os
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
import pandas as pd
import sys
from scipy.fft import fft, ifft


def psd_plot(freq, Y):
    plt.xlim([0, 50])
    plt.plot(freq, pow(abs(Y), 2))
    plt.show()



if __name__ == '__main__':
    deltafreq = [1, 3]
    thetafreq = [4, 8]
    alphafreq = [8, 14]
    betafreq = [14, 31]
    gammafreq = [31, 50]

    folder_ = "result"
    fs = 500
    window = "hamming"  # boxcar, hamming, hann
    n = 5000

    # df = pd.DataFrame(columns=)
    fold_list = os.listdir(folder_)
    for j, file in enumerate(fold_list):  # 파일 개수에 따른.
        if 'csv' in file:
            x = pd.read_csv(folder_ + '/' + file)
            x = x[3000:]
            x = x.drop(['Unnamed: 0'], axis='columns')
            x.reset_index(drop=True, inplace=True)
            df_beta = pd.DataFrame(index=range(1, 6), columns=x.columns)
            # df_theta = pd.DataFrame(index=range(1, 6), columns=x.columns)
            # df_alpha = pd.DataFrame(index=range(1, 6), columns=x.columns)

            for i in range(len(x.columns)):  # 각 채널들 6개.
                ch = x.iloc[:, i]
                ch = np.array(ch)
                for k in range(0, int(len(ch)/n)): #len(ch) = 25000
                    T = n / fs
                    kn = np.arange(n)
                    freq = kn / T
                    Y = np.fft.fft(ch[n*k:n*(k+1)]) / n  # fft computing and normalization
                    freq = freq[range(int(n / 2))]
                    Y = Y[range(int(n / 2))]
                    Y = pow(abs(Y), 2)
                    psd_plot(freq, Y)
                    # beta_y = Y[alphafreq[0] * 10: alphafreq[1] * 10]
                    #
                    # # beta_y = Y[betafreq[0] * 10: betafreq[1] * 10]
                    # # theta_y = Y[thetafreq[0] * 10: thetafreq[1] * 10]
                    # # alpha_y = Y[alphafreq[0] * 10: alphafreq[1] * 10]
                    #
                    # beta_mean = beta_y.mean() * 10 ** 8  # 10^-8
                    # # theta_mean = theta_y.mean() * 10 ** 8
                    # # alpha_mean = alpha_y.mean() * 10 ** 8
                    #
                    # df_beta.iloc[k, i] = beta_mean
                    # df_theta.iloc[k, i] = df_theta
                    # df_alpha.iloc[k, i] = df_alpha
                    break
            # print(df_beta)
            df_beta.to_csv(f'result/psd/{file[:-4]}alpha.csv', index=False)
            # df_theta.to_csv(f'result/psd/{file[:-4]}theta.csv', index=False)
            # df_alpha.to_csv(f'result/psd/{file[:-4]}alpha.csv', index=False)

