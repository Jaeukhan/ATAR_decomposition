import numpy as np
import matplotlib.pyplot as plt
import spkit as sp
from spkit.data import load_data
import os
import pandas as pd

folder_ = "cutting"
fold_list = os.listdir(folder_)
print(fold_list)
def cut_add(df2, df, first, last):
    df2 = pd.concat([df2, df[first:last]])
    return df2


for i, file in enumerate(fold_list):
    noise = pd.DataFrame(columns=["AF3","AF4","Fp1","Fp2","AF7","AF8"])
    df = pd.read_csv(folder_+'/'+file)
    df = df.drop(['Unnamed: 0'], axis='columns')

    for j in range(0,len(df), 2500):
        noise = cut_add(noise, df, j+1750, j+2150)#3.5초 ~ 4.3초 400개
    # noise = cut_add(noise, df, 1,250)
    noise.reset_index(drop=True, inplace=True)
    noise.to_csv(f'result/stp{fold_list[i][3:-4]}.csv', index=True)
