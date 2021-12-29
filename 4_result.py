import numpy as np
import matplotlib.pyplot as plt
# from scipy import signal
# from joblib import Parallel, delayed
import spkit as sp
from spkit.data import load_data
import os
import pandas as pd

fold = ["12su", "12sb", "12kh", "12dh"]
outlier = False

for j in range(len(fold)):
    fold_list = os.listdir("concat/%s" % fold[j])
    print(fold_list)
    li = []
    for i, file in enumerate(fold_list):  #
        df = pd.read_csv("concat/" + fold[j] + '/' + file)
        li.append(df["poall"].values)
        # if (i+1) % 5 == 0:
        #     k += 1
        #     col = 0
        # print(df['poall'].values)
        # # df2.append(df["poall"].values)
        # df2.iloc[4 * k:4 * (k + 1), col] = df["poall"].values
        # col+= 1
        # df2.to_csv(f'result/{fold_list[i][:-7]}.csv', index=False)
        # k += 1
    df2 = pd.DataFrame(li, columns=["delta", "theta", "alpha", "beta"])
    if outlier:
        for i in range(len(df2), 0, -1):
            df2.loc[df2.delta > 30, :] = 0
        df2.to_excel(f'result/xout{fold[j]}.xlsx', index=False)
    df2.to_excel(f'result/{fold[j]}.xlsx', index=False)
