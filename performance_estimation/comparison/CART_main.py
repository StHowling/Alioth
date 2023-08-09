# %%
import os
import nni
import csv
import random
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from easydict import EasyDict
from model import LinearRegression, LRG_random_dataset, CART_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
# %%
# setting up logger

# logfile = "./log/CART.log"
# make_file_dir(logfile)
# logger = logging.getLogger("CART logger")
# logger.setLevel(logging.DEBUG)
# ch = logging.FileHandler(logfile)
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
# ch.setFormatter(formatter)
# logger.addHandler(ch)
# %%

app_clusters = {
    '0':['hbase','cassandra','mongoDB'],
    '1':['kafka','rabbitmq'],
    '2':['etcd','redis'],
    '3':['milc'],
}


DIR = "/home/yyx/interference_prediction/Alioth/DATA/TOT+_af_fs.csv"
data = pd.read_csv(DIR)

# CLUSTER_0 = raw_data
# # CLUSTER_0 = raw_data[(raw_data['APP'] == 'redis' )]
# CLUSTER_0 = CLUSTER_0.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)
# print(CLUSTER_0)

# CLUSTER_0 = np.array(CLUSTER_0, dtype = 'float32')

# XGBOOST_DATA_0 = CLUSTER_0[:,0:218]
# LABEL_0 = CLUSTER_0[:,218]


data = data.drop(['Unnamed: 0'], axis=1)
# data = data[(data['APP'] == 'redis' )]
# data = data.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)
print(data)
label_data = data['QoS']
data = data.drop(['QoS'], axis=1)
VM_data = data
VM_data = np.array(VM_data, dtype = 'float32')[:,0:152]
print(np.shape(VM_data))
label_data = np.array(label_data, dtype = 'float32')



train_dataset, test_dataset, train_label, test_label = train_test_split(VM_data,label_data,test_size=0.2,random_state=1)


clf = tree.DecisionTreeRegressor()
clf = clf.fit(train_dataset, train_label)
predict = clf.predict(test_dataset)


#设置阈值判断劣化强度
label_1 = 1.05 # >1.05开始劣化  1.05~1.15轻微劣化
label_2 = 1.25 #>1.25强劣化
label_3 = 1.15 #1.15~1.25中等劣化

#计数劣化、强劣化
j=k=0#假阳性
m=n=0#真阳性
p=q=0#FN
#计数轻微列优化、中等劣化
tp_mid = tp_min = 0
fp_mid = fp_min = 0
fn_mid = fn_min = 0

for i in range(len(predict)):
    if predict[i]> label_1 and test_label[i]< label_1:
        j+=1
    if predict[i]> label_2 and test_label[i]< label_2:
        k+=1
    if predict[i]> label_1 and test_label[i]> label_1:
        m+=1
    if predict[i]> label_2 and test_label[i]> label_2:
        n+=1
    if predict[i]< label_1 and test_label[i]> label_1:
        p+=1
    if predict[i]< label_2 and test_label[i]> label_2:
        q+=1
    if predict[i]> label_3 and predict[i]< label_2 and test_label[i]> label_3 and test_label[i]< label_2:
        tp_mid+=1
    if predict[i]> label_3 and predict[i]< label_2 and (test_label[i]< label_3 or test_label[i]> label_2):
        fp_mid+=1  
    if (predict[i]< label_3 or predict[i]> label_2) and test_label[i]> label_3 and test_label[i]< label_2:
        fn_mid+=1 
    if predict[i]> label_1 and predict[i]< label_3 and test_label[i]> label_1 and test_label[i]< label_3:
        tp_min+=1
    if predict[i]> label_1 and predict[i]< label_3 and (test_label[i]< label_1 or test_label[i]> label_3):
        fp_min+=1  
    if (predict[i]< label_1 or predict[i]> label_3) and test_label[i]> label_1 and test_label[i]< label_3:
        fn_min+=1    
    


print("是否发生劣化 Precision_1.05:",m/(j+m))
print("是否是强劣化 Precision_1.25:",n/(k+n))
print("是否发生劣化 Recall_1.05:",m/(p+m))
print("是否是强劣化 Recall_1.25:",n/(q+n))

print("是否发生中等劣化 Precision:", tp_mid/(tp_mid+fp_mid))
print("是否是轻微劣化 Precision:", tp_min/(tp_min+fp_min ))
print("是否发生中等劣化 Recall:", tp_mid/(tp_mid+fn_mid))
print("是否是轻微劣化 Recall:", tp_min/(tp_min+fn_min))


print("MSE",metrics.mean_squared_error(predict, test_label))
print("MAE",metrics.mean_absolute_error(predict, test_label))
print("MAPE",metrics.mean_absolute_percentage_error(predict, test_label))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True



# %%
# Get and update args
setup_seed(146)
# s