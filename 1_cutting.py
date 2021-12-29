import numpy as np
import matplotlib.pyplot as plt
#from scipy import signal
#from joblib import Parallel, delayed
import spkit as sp
from spkit.data import load_data
import os
import pandas as pd

folder_ = "1215su" #12.09/su
fold_list = os.listdir("WaveResults/%s" % folder_)
print(fold_list)
step_size = 30000

start = 1750 #500
end = 1500 #-1500

if not os.path.isdir(f'cutting/{folder_}'):
    os.mkdir(f'cutting/{folder_}')

for i, file in enumerate(fold_list):
    df = pd.read_csv("WaveResults/"+folder_+'/'+file)
    df = df[start:]
    # print(df)
    X = df.drop(['Unnamed: 0'], axis='columns')
    X.reset_index(drop=True, inplace=True)
    X.to_csv(f'cutting/{folder_}/cut{fold_list[i][:-4]}.csv', index=True)

    """3 partion"""
    # X1 = X[step_size * 0:step_size * 1]
    # X2 = X[step_size * 1:step_size * 2]
    # X3 = X[step_size * 2:step_size * 3]
    #
    # X1.to_csv(f'cutting/cut{3}_{fold_list[i][:-4]}.csv', index=True)
    # X2.to_csv(f'cutting/cut{6}_{fold_list[i][:-4]}.csv', index=True)
    # X3.to_csv(f'cutting/cut{9}_{fold_list[i][:-4]}.csv', index=True)