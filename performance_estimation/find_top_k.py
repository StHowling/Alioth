import numpy as np
import pandas as pd


print('---------------------------workload type -------------------------------')

DIR = "/home/yyx/interference_prediction/Alioth/DATA/stress_part_cor.csv"
DATA = pd.read_csv(DIR)
print(DATA)
NAME = DATA['Unnamed: 0']
APP = [column for column in DATA]
print(APP[1:])

line = 0.4
L = []

for i in range(218):
    for j in APP[1:]:
        if DATA[j][i] >= line or DATA[j][i] <= -line:
            L.append(NAME[i])
L = list(set(L))
print(len(L))
L = pd.DataFrame(L)
L.to_csv("/home/yyx/interference_prediction/Alioth/DATA/28_LABEL_app.csv")


print('---------------------------stress type -------------------------------')


# dir = "/home/yyx/interference_prediction/Alioth/DATA/correlation.csv"
# data = pd.read_csv(dir)
# print(data)
# name = data['Unnamed: 0']
# Stress = [column for column in data]
# print(Stress[1:])

# # line = 0.35
# LL = []

# for i in range(218):
#     for j in Stress[1:]:
#         if data[j][i] >= line or data[j][i] <= -line:
#             LL.append(name[i])
# print(len(list(set(LL))))


print('---------------------------tot -------------------------------')

# print(len(list(set(LL))+list(set(L))))
