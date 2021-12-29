import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from joblib import Parallel, delayed
import spkit as sp
from spkit.data import load_data
import os
import pandas as pd

folder_ = "res"
fold_list = os.listdir("WaveResults/%s" % folder_)
print(fold_list)
"""mutual information"""

"""
folder_ = "11.12"
fold_list = os.listdir("WaveResults/%s" % folder_)

file_path = ""
for (root, directories, files) in os.walk(f"WaveResults/{folder_}/" + fold_list[0]):
    for file in files:
        if '.csv' in file:
            file_path = os.path.join(root, file)


savename = "noise"
file_path = "result/dh1_elim_0.6.csv"
df = pd.read_csv(file_path)
X = df[7000:]
# X = df.drop(['Unnamed: 0'], axis='columns')
# X.reset_index(drop=True, inplace=True)
ch_names = list(X.columns)
X = np.array(X)
fs = 128

t = np.arange(X.shape[0])/128
nC = len(ch_names)

MI = np.zeros([nC, nC])

for i in range(nC):
    x1 = X[:, i]
    for j in range(nC):
        x2 = X[:, j]

        # Mutual Information
        MI[i, j] = sp.mutual_Info(x1, x2)

print(MI.mean())
"""


def cut_add(df2, df, first, last):
    df2 = pd.concat([df2, df[first:last]])
    return df2

for i, file in enumerate(fold_list):
    df2 = pd.DataFrame(columns=["AF3","AF4","Fp1","Fp2","AF7","AF8"])
    noise = df2 = pd.DataFrame(columns=["AF3","AF4","Fp1","Fp2","AF7","AF8"])

    df = pd.read_csv("WaveResults/"+folder_+'/'+file)
    df = df.drop(['Unnamed: 0'], axis='columns')
    print(df2)
    # df2 = pd.concat([df2, df[251:2250]])
    # df2 = pd.concat([df2,df[3501:4000]])
    # df2 = pd.concat([df2,df[4751:5500]])
    # df2 = pd.concat([df2,df[6501:7000]])
    # df2 = pd.concat([df2,df[7751:8500]])
    # df2 = pd.concat([df2,df[9501:10000]])
    # df2 = pd.concat([df2,df[10751:11250]])
    # df2 = pd.concat([df2,df[12251:13000]])
    # df2 = pd.concat([df2,df[13251:13500]])
    # df2 = pd.concat([df2,df[14251:17500]])
    noise = cut_add(noise, df, 1,250)
    noise = cut_add(noise, df, 751, 2000)
    noise = cut_add(noise, df, 3501, 6000)
    # print(df)
    df2.reset_index(drop=True, inplace=True)

    noise.to_csv(f'result/{fold_list[i][:-4]}.csv', index=True)
#
# file_path = "WaveResults/11.24/ju/walking.csv"
# fs = 500
#
# df = pd.read_csv(file_path)
# df = df[3000:]
# # print(df)
# X = df.drop(['Unnamed: 0'], axis='columns')
# X.reset_index(drop=True, inplace=True)
#
# X.to_csv(f'result/walking_MiCC.csv', index=True)

