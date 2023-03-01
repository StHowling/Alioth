import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost
import shap
import itertools as IT
import operator



dir = "/home/yyx/interference_prediction/Alioth/DATA/28_LABEL.csv"
LABEL = pd.read_csv(dir)
L = list(LABEL['LABEL'])
print(L)
DIR = "/home/yyx/interference_prediction/Alioth/DATA/correlation.csv"
DATA = pd.read_csv(DIR)
DATA.columns = ['NAME','FIO','L','MBW','NET']
# DATA.columns =['NAME','cassandra', 'etcd', 'hbase', 'kafka', 'milc', 'mongoDB', 'rabbitmq', 'redis']

MATRIX = DATA.loc[DATA['NAME'].isin(L)]
MATRIX.drop(['NAME'], axis=1,inplace = True)
DATA.drop(['NAME'], axis=1,inplace = True)
# print('DATA',DATA)
for col in DATA.columns:
    DATA[col].values[:] = 0
# print(DATA)
MATRIX = DATA + MATRIX
MATRIX.fillna(0,inplace = True)
MATRIX = np.array(MATRIX)
# print(np.shape(MATRIX))

##########
dirpre1 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predmilc.npy'
VM_data_no_stress = np.load(dirpre1, allow_pickle=True)

dirpre2 = "/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv"
raw_data = pd.read_csv(dirpre2)
data = raw_data.loc[(raw_data['APP'] == 'milc' )]
STRESS = data['stress_type']
data = data.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)

data = np.array(data,dtype = 'float32')

VM_data = data[:,0:218]
label_data = data[:,218]

VM_data = np.append(VM_data,VM_data_no_stress, axis=1)
print(np.shape(VM_data))

# train_dataset, test_dataset, train_label, test_label = train_test_split(VM_data,label_data,test_size=0.8,random_state=1)

######设置测试数据
# inVM = test_dataset[2,:].reshape((1,436))
# inQOS = test_label[2]

model = xgb.XGBRegressor( eta=0.01, colsample_bylevel=1, colsample_bytree = 0.8, learning_rate=0.03, max_depth=6, n_estimators=1000).fit(VM_data,label_data)
predict = model.predict(VM_data)


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

explainer = shap.Explainer(model)
shap_values = explainer.shap_values(VM_data)
# np.save("/home/yyx/interference_prediction/Alioth/Result/xgboost_shap_values_SOI.npy",shap_values)
global_shap_values = pd.DataFrame(np.abs(shap_values).mean(0)).reset_index()
global_shap_values.columns = ["var","feature_importance"]
# np.save("/home/yyx/interference_prediction/Alioth/Result/global_shap_values_SOI.npy",global_shap_values)

print("shap_values:", np.shape(shap_values[:,0:218]))
# print("global_shap_values:",global_shap_values)

S1 = shap_values[:,0:218]
S2 = shap_values[:,218:436]

C1 = np.matmul(S1,MATRIX)
C2 = np.matmul(S2,MATRIX)

C = C1 + C2
C = C/np.sum(C)
C = pd.DataFrame(C)
C.columns = ["FIO", "L", "MBW", "NET"]
result = pd.concat([C,STRESS], axis = 1)
result.columns = ["FIO", "L", "MBW", "NET","stress_type"]
result.to_csv("/home/yyx/interference_prediction/Alioth/DATA/find_SOI.csv")

rowmax = C.max(axis=1)
C.values == rowmax[:,None]
id = np.where(C.values == rowmax[:,None])
groups = IT.groupby(zip(*id), key=operator.itemgetter(0))

find_soi_true = 0
stess_pred = [[C.columns[j] for i,j in grp] for k,grp in groups]
STRESS = list(STRESS)

deg=0
for i in range(len(STRESS)):
    if predict[i]>=1.05:
        deg +=1
        if STRESS[i] == stess_pred[i][0]:
            find_soi_true += 1 
print("deg:", deg)
print("percentage:", find_soi_true/deg)
