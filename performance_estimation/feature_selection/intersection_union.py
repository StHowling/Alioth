import numpy as np
import pandas as pd

#%%
# DIR = "/home/yyx/interference_prediction/Alioth/Result/global_shap_values_milc.npy"
# data = np.load(DIR, allow_pickle=True)

# data_shap = pd.DataFrame(data) 
# data_shap.columns = ["var","feature_importance"]
# data_shap = data_shap.sort_values(by=['feature_importance'],ascending=False)
# print("data_shap",data_shap)
# print(data_shap.iloc[:40])

# data_shap.iloc[:40].to_csv("/home/yyx/interference_prediction/Alioth/Result/xgb_fs40_milc.csv")



#%%
DIR1 =  "/home/yyx/interference_prediction/Alioth/Result/SORT_fs40.csv"
DIR2 = "/home/yyx/interference_prediction/Alioth/Result/LABEL_NAME.csv"

SORT_fs40 = pd.read_csv(DIR1)
LABEL = pd.read_csv(DIR2, header = None)
LABEL.columns = ["NAME"]
print(SORT_fs40)

SORT_fs40["var"]
L=[]
NAMEL = []
for i in range(40):
    L.append(SORT_fs40["var"][i])
    L.append(SORT_fs40["var.1"][i])
    L.append(SORT_fs40["var.2"][i])
    L.append(SORT_fs40["var.3"][i])
    L.append(SORT_fs40["var.4"][i])
    L.append(SORT_fs40["var.5"][i])
    L.append(SORT_fs40["var.6"][i])
    L.append(SORT_fs40["var.7"][i])

L = np.unique(L).tolist()
# print(L)
print("len",len(L))
for i in L:
    NAMEL.append(LABEL["NAME"][i])
print(NAMEL)

NAMEL = pd.DataFrame(NAMEL)
NAMEL.columns = ["NAME"]
print(NAMEL)
TOT = pd.DataFrame(L)
TOT.columns = ["var"]
TOT = pd.concat([TOT,NAMEL],axis = 1)
print(TOT)

 
TOT.to_csv("/home/yyx/interference_prediction/Alioth/Result/fsforall.csv")


set0 = set(SORT_fs40["var"].tolist())
set1 = set(SORT_fs40["var.1"].tolist())
set2 = set(SORT_fs40["var.2"].tolist())
set3 = set(SORT_fs40["var.3"].tolist())
set4 = set(SORT_fs40["var.4"].tolist())
set5 = set(SORT_fs40["var.5"].tolist())
set6 = set(SORT_fs40["var.6"].tolist())
set7 = set(SORT_fs40["var.7"].tolist())

SAME = set0&set1&set2&set3&set4&set5&set6&set7
SAME = list(SAME)
NAMESAME = []
for j in SAME:
    NAMESAME.append(LABEL["NAME"][j])
print(NAMESAME)


# print("cosine_similarity(data_shap, data_FI):",cosine_similarity(SORT_fs40["var"], SORT_fs40["var.1"]))