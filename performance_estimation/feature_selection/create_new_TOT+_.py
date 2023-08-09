import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from  torch.nn import init

DIR = "/home/yyx/interference_prediction/Alioth/Result/fsforall.csv"
DATA_FEATURE = pd.read_csv(DIR)

FEATURE = list(DATA_FEATURE["NAME"])
FEATURE_number = list(DATA_FEATURE["var"])

Remove_number = []
for i in range(218):
    if i not in FEATURE_number:
        Remove_number.append(i)
print(Remove_number)

#过滤指标筛选结果
DIR1 = "/home/yyx/interference_prediction/Alioth/DATA/TOT+_.csv"
data = pd.read_csv(DIR1)
data = data.drop(['Unnamed: 0'], axis=1)
print(data)

DIR2 = "/home/yyx/interference_prediction/Alioth/Result/LABEL_NAME.csv"
LABEL = pd.read_csv(DIR2, header = None)
LABEL.columns = ["NAME"]
LABEL = list(LABEL["NAME"]) 

for i in FEATURE:
    LABEL.remove(i)
print(LABEL)

print(len(LABEL))

for i in LABEL:
    data = data.drop([i], axis=1)
    data = data.drop(['Label_'+ i], axis=1)   
print(data)

data.to_csv("/home/yyx/interference_prediction/Alioth/DATA/TOT+_af_fs.csv")


#对DAE输出 过滤指标筛选结果
dirpre = '/home/yyx/interference_prediction/Alioth/Result/DAE_PredTOT.npy'
VM_data_no_stress = np.load(dirpre, allow_pickle=True)
print(np.shape(VM_data_no_stress))

VM_data_no_stress_af_fs = np.delete(VM_data_no_stress, Remove_number, axis = 1)
print(np.shape(VM_data_no_stress_af_fs))
np.save("/home/yyx/interference_prediction/Alioth/DATA/DAE_TOT+_af_fs.npy",VM_data_no_stress_af_fs)
