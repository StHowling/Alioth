# %%
import os
import sys
import logging
import numpy as np
import pandas as pd
import pickle
import json
import random
import matplotlib.pyplot as plt
from easydict import EasyDict

# %%
# dir1 = "/home/yyx/DP/classified11/1_data.csv"
# dir2 = "/home/yyx/DP/classified2/1_data.csv"

# df1 = pd.DataFrame(pd.read_csv(dir1))
# print(df1.shape)
# df2 = pd.DataFrame(pd.read_csv(dir2))
# print(df2.shape)
# result = pd.concat([df1,df2],sort = False)
# print(result)
# result.to_csv("/home/yyx/DP/classified_total/1_data_NEW.csv")

#%%
DIR = "/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv"
DATA = pd.read_csv(DIR)
print(DATA["QoS"].shape)
print(DATA["QoS"])
j=0

for i in range(6904):
    if DATA["QoS"][i]<0.95:
        DATA["QoS"][i] = 0.95

for i in range(6904):
    if DATA["QoS"][i]< 0.95:
        j+=1

DATA = DATA.sort_values(by=['APP'],ascending=True)
DATA = DATA.sort_values(by=['stress_type'],ascending=False)
DATA = DATA.reset_index(drop = True)
print("num of < 0.95:", j)
print(DATA)




# L = [i for i in range(j)]
# DROP = random.sample(L,j-656)
# print(DROP)
# RESULT = DATA.drop(DROP,)
# print(RESULT["QoS"].shape)
# print(RESULT["QoS"])
# DATA.to_csv("/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv")
