import os
import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
import pandas as pd
from collections import Counter

def detect_outliers(df, n, features): #Outliers_to_drop = detect_outliers(df_train, 2, ["변수명"])
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers



def atar_plotting(folder_, optmode="elim", beta=0.6, ipr=[25, 75], wv="db3", plotting=False, cut= True):
    fold_list = os.listdir("cutting/%s" % folder_)
    print(fold_list)

    for i, file in enumerate(fold_list):
        df = pd.read_csv("cutting/%s" % folder_+'/'+file)
        if cut==True:
            df = df[3000:]
        # print(df)
        X = df.drop(['Fp2','Unnamed: 0'], axis='columns')
        X.reset_index(drop=True, inplace=True)
        ch_names = list(X.columns)
        X = np.array(X)
        fs = 500
        # filter with highpass
        Xf = sp.filter_X(X, band=[1], btype='highpass', fs=fs, verbose=0).T
        Xf = Xf * 500
        t = np.arange(Xf.shape[0]) / fs

        XR = sp.eeg.ATAR_mCh_noParallel(Xf.copy(), wv=wv, verbose=0, beta=beta, OptMode=optmode, IPR=ipr)

        if plotting != True:
            pass
        else:
            plt.figure(figsize=(20, 5))
            plt.plot(t, Xf + np.arange(-3, 3) * 200)
            plt.xlim([t[0], t[-1]])
            plt.xlabel('time (sec)')
            plt.yticks(np.arange(-3, 3) * 200, ch_names)
            plt.grid()
            plt.title('Xf: 14 channel - EEG Signal (filtered)')
            plt.show()

            plt.figure(figsize=(15, 5))
            plt.subplot(121)
            plt.plot(t, XR + np.arange(-3, 3) * 200)
            plt.xlim([t[0], t[-1]])
            plt.xlabel('time (sec)')
            plt.yticks(np.arange(-3, 3) * 200, ch_names)
            plt.grid()
            plt.title('XR: Corrected Signal: ' + r'$\beta=$' + f'{beta}')

            plt.subplot(122)
            plt.plot(t, (Xf - XR) + np.arange(-3, 3) * 200)
            plt.xlim([t[0], t[-1]])
            plt.xlabel('time (sec)')
            plt.yticks(np.arange(-3, 3) * 200, ch_names)
            plt.grid()
            plt.title('Xf - XR: Difference (removed signal)')

        df2 = pd.DataFrame(XR / 500, columns=ch_names)
        # plt.savefig(f'plot/{fold_list[j]}{i + 5}_{optmode}_{int(beta * 10)}.png', dpi=300)
        if plotting != True:
            pass
        else:
            plt.show()
        df2.to_csv(f'result/{folder_}/at{fold_list[i][:-4]}.csv', index=True)
    # for j in range(len(fold_list)):
    #     for (root, directories, files) in os.walk(f"WaveResults/{folder_}/" + fold_list[j]):
    #         for i, file in enumerate(files):
    #             file_path = os.path.join(root, file)
    #
    #             df = pd.read_csv(file_path)
    #             # df = df[3000:]
    #             # print(df)
    #             X = df.drop(['Unnamed: 0'], axis='columns')
    #             X.reset_index(drop=True, inplace=True)
    #             ch_names = list(X.columns)
    #             X = np.array(X)
    #             fs = 500
    #             # filter with highpass
    #             Xf = sp.filter_X(X, band=[1], btype='highpass', fs=fs, verbose=0).T
    #             Xf = Xf * 500
    #             t = np.arange(Xf.shape[0]) / fs
    #             if plotting != True:
    #                 pass
    #             else:
    #                 plt.figure(figsize=(20, 5))
    #                 plt.plot(t, Xf + np.arange(-3, 3) * 200)
    #                 plt.xlim([t[0], t[-1]])
    #                 plt.xlabel('time (sec)')
    #                 plt.yticks(np.arange(-3, 3) * 200, ch_names)
    #                 plt.grid()
    #                 plt.title('Xf: 14 channel - EEG Signal (filtered)')
    #                 plt.show()
    #
    #             XR = sp.eeg.ATAR_mCh_noParallel(Xf.copy(), wv=wv, verbose=0, beta=beta, OptMode=optmode, IPR=ipr)
    #
    #             plt.figure(figsize=(15, 5))
    #             plt.subplot(121)
    #             plt.plot(t, XR + np.arange(-3, 3) * 200)
    #             plt.xlim([t[0], t[-1]])
    #             plt.xlabel('time (sec)')
    #             plt.yticks(np.arange(-3, 3) * 200, ch_names)
    #             plt.grid()
    #             plt.title('XR: Corrected Signal: ' + r'$\beta=$' + f'{beta}')
    #
    #             plt.subplot(122)
    #             plt.plot(t, (Xf - XR) + np.arange(-3, 3) * 200)
    #             plt.xlim([t[0], t[-1]])
    #             plt.xlabel('time (sec)')
    #             plt.yticks(np.arange(-3, 3) * 200, ch_names)
    #             plt.grid()
    #             plt.title('Xf - XR: Difference (removed signal)')
    #
    #             df2 = pd.DataFrame(XR/500, columns=ch_names)
    #             plt.savefig(f'plot/{fold_list[j]}{i + 5}_{optmode}_{int(beta*10)}.png', dpi=300)
    #             if plotting != True:
    #                 pass
    #             else:
    #                 plt.show()
    #             df2.to_csv(f'result/at{fold_list[i][3:-4]}.csv', index=True)
                # df2.to_csv(f'result/{fold_list[j]}{i+5}_{optmode}_{int(beta*10)}.csv', index=True)

def tuning_beta(folder_, optmode="elim"):
    """
    :param folder_: data folder name
    :param optmode: soft, linAtten, elim
    :return: no return & save png
    """
    fold_list = os.listdir("WaveResults/%s" % folder_)

    file_path = ""
    for (root, directories, files) in os.walk(f"WaveResults/{folder_}/" + fold_list[0]):
        for file in files:
            if '.csv' in file:
                file_path = os.path.join(root, file)

    df = pd.read_csv(file_path)
    df = df[7000:]
    X = df.drop(['Unnamed: 0'], axis='columns')
    X.reset_index(drop=True, inplace=True)
    ch_names = list(X.columns)
    X = np.array(X)
    fs = 500
    # filter with highpass
    Xf = sp.filter_X(X, band=[1], btype='highpass', fs=fs, verbose=0).T
    Xf = Xf * 500

    nC = len(ch_names)
    li = []
    t = np.arange(Xf.shape[0]) / fs
    betas = np.r_[np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.1)].round(2)

    for b in betas:
        MI = np.zeros([nC, nC])
        XR = sp.eeg.ATAR_mCh_noParallel(Xf.copy(), wv='db3', verbose=0, beta=b, OptMode=optmode, IPR=[15, 85])
        for i in range(nC):
            x1 = XR[:, i]
            for j in range(nC):
                x2 = XR[:, j]

                # Mutual Information
                MI[i, j] = sp.mutual_Info(x1, x2)
        li.append(MI.mean())
    li = np.array(li)
    print(li.argmax())
    ord = li.argmax()
    print(li[ord])

def tuning_beta_plot(folder_, optmode="elim"):
    """
    :param folder_: data folder name
    :param optmode: soft, linAtten, elim
    :return: no return & save png
    """
    fold_list = os.listdir("WaveResults/%s" % folder_)

    file_path = ""
    for (root, directories, files) in os.walk(f"WaveResults/{folder_}/" + fold_list[0]):
        for file in files:
            if '.csv' in file:
                file_path = os.path.join(root, file)

    df = pd.read_csv(file_path)
    df = df[7000:]
    X = df.drop(['Unnamed: 0'], axis='columns')
    X.reset_index(drop=True, inplace=True)
    ch_names = list(X.columns)
    X = np.array(X)
    fs = 500
    # filter with highpass
    Xf = sp.filter_X(X, band=[1], btype='highpass', fs=fs, verbose=0).T
    Xf = Xf * 500

    t = np.arange(Xf.shape[0]) / fs
    betas = np.r_[np.arange(0.01, 0.1, 0.02), np.arange(0.1, 1, 0.1)].round(2)
    for b in betas:
        XR = sp.eeg.ATAR_mCh_noParallel(Xf.copy(), wv='db3', verbose=0, beta=b, OptMode=optmode, IPR=[25, 75])

        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(t, XR + np.arange(-3, 3) * 200)
        plt.xlim([t[0], t[-1]])
        plt.xlabel('time (sec)')
        plt.yticks(np.arange(-3, 3) * 200, ch_names)
        plt.grid()
        plt.title('XR: Corrected Signal: ' + r'$\beta=$' + f'{b}')

        plt.subplot(122)
        plt.plot(t, (Xf - XR) + np.arange(-3, 3) * 200)
        plt.xlim([t[0], t[-1]])
        plt.xlabel('time (sec)')
        plt.yticks(np.arange(-3, 3) * 200, ch_names)
        plt.grid()
        plt.title('Xf - XR: Difference (removed signal)')
        plt.savefig(f'plot/{optmode}_db3_{b}.png', dpi=300)