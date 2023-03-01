import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics


app_clusters = {
    '0':['hbase','cassandra','mongoDB'],
    '1':['kafka','rabbitmq'],
    '2':['etcd','redis'],
    '3':['milc'],
}


# DIR = "/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv"
# raw_data = pd.read_csv(DIR)

# CLUSTER_0 = raw_data
# # CLUSTER_0 = raw_data[(raw_data['APP'] == 'mongoDB' )]

# CLUSTER_0 = CLUSTER_0.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)
# print(CLUSTER_0)

# CLUSTER_0 = np.array(CLUSTER_0, dtype = 'float32')


# XGBOOST_DATA_0 = CLUSTER_0[:,0:218]
# LABEL_0 = CLUSTER_0[:,218]
# print(LABEL_0)

# train_dataset_0, test_dataset_0, train_label_0, test_label_0 = train_test_split(XGBOOST_DATA_0,LABEL_0,test_size=0.2,random_state=1)

# train_dataset = train_dataset_0
# train_label = train_label_0
# test_dataset = test_dataset_0
# test_label = test_label_0

# print(len(CLUSTER_0))
# # print("kafka")




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


'''
XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints=None,
       learning_rate=0.300000012, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints=None,
       n_estimators=100, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=1, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
       validate_parameters=False, verbosity=None)
'''


#查看特征的重要程度
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import xgboost
import shap

# train an XGBoost model
model = xgb.XGBRegressor(n_estimators=100).fit(train_dataset, train_label)
predict = model.predict(test_dataset)
# print("predict",predict)
# print('labe',test_label)
_ = xgb.plot_importance(model,height = 0.9)
FI = model.feature_importances_


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


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

# explainer = shap.Explainer(model)

# shap_values = explainer.shap_values(test_dataset)
# np.save("/home/yyx/interference_prediction/Alioth/Result/xgboost_shap_values_rabbitmq.npy",shap_values)
# global_shap_values = pd.DataFrame(np.abs(shap_values).mean(0)).reset_index()
# global_shap_values.columns = ["var","feature_importance"]
# np.save("/home/yyx/interference_prediction/Alioth/Result/global_shap_values_mongoDB.npy",global_shap_values)
# np.save("/home/yyx/interference_prediction/Alioth/Result/xgboost_feature_importance_redis.npy",FI)


