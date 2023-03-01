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


# %%
#数据
dirpre1 = '/home/yyx/interference_prediction/Alioth/DATA/DAE_Pred-redis.npy'
VM_data_no_stress = np.load(dirpre1, allow_pickle=True)

dirpre1 = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predredis.npy'
VM_data_no_stress_test = np.load(dirpre1, allow_pickle=True)


DIR = "/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv"
raw_data = pd.read_csv(DIR)
data = raw_data.loc[(raw_data['APP'] != 'redis' )]
data = data.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)
print(data)
data_test = raw_data[(raw_data['APP'] == 'redis' )]
data_test = data_test.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','Unnamed: 0'], axis=1)
print(data_test)

data = np.array(data,dtype = 'float32')
# data_test = np.array(data_test,dtype = 'float32')

VM_data = data[:,0:218]
label_data = data[:,218]
# VM_data_test = data_test[:,0:218]
# label_data_test = data_test[:,218]

# VM_data = VM_data_stress - VM_data_no_stress 
VM_data = np.append(VM_data,VM_data_no_stress, axis=1)
VM_data_test = np.append(VM_data_test,VM_data_no_stress_test, axis=1)
print(np.shape(VM_data))


train_dataset, test_dataset, train_label, test_label = train_test_split(VM_data,label_data,test_size=0.2,random_state=1)
train_dataset_1, test_dataset_1, train_label_1, test_label_1 = train_test_split(VM_data_test,label_data_test,test_size=0.2,random_state=1)

# print(train_label_1)
# print(test_dataset_1)
'''
XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
       importance_type='gain', interaction_constraints=None,
       learning_rate=0.300000012, max_delta_step=0, max_depth=6,
       min_child_weight=1, missing=nan, monotone_constraints=None,
       n_estimators=100, n_jobs=0, num_parallel_tree=1,
       objective='reg:squarederror', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
       validate_parameters=False, verbosity=None)
'''
# train an XGBoost model
model = xgb.XGBRegressor( eta=0.01, colsample_bylevel=0.8, colsample_bytree = 0.6, learning_rate=0.03, max_depth=6, n_estimators=1000).fit(train_dataset, train_label)
predict = model.predict(test_dataset_1)
print("predict",predict)
print("test",test_label_1)
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

test_label = test_label_1

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
# print("是否是轻微劣化 Precision:", tp_min/(tp_min+fp_min))
print("是否发生中等劣化 Recall:", tp_mid/(tp_mid+fn_mid))
# print("是否是轻微劣化 Recall:", tp_min/(tp_min+fn_min))




print("MSE",metrics.mean_squared_error(predict, test_label))
print("MAE",metrics.mean_absolute_error(predict, test_label))
print("MAPE",metrics.mean_absolute_percentage_error(predict, test_label))


# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)

# explainer = shap.Explainer(model)
# shap_values = explainer.shap_values(test_dataset)
# np.save("/home/yyx/interference_prediction/Alioth/Result/xgboost_shap_values_0.npy",shap_values)
# global_shap_values = pd.DataFrame(np.abs(shap_values).mean(0)).reset_index()
# global_shap_values.columns = ["var","feature_importance"]
# np.save("/home/yyx/interference_prediction/Alioth/Result/global_shap_values_0.npy",global_shap_values)
# np.save("/home/yyx/interference_prediction/Alioth/Result/xgboost_feature_importance_0.npy",FI)


# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
# shap.plots.force(shap_values[0])
# shap.plots.force(explainer.expected_value, shap_values)
# shap.plots.beeswarm(explainer.expected_value, shap_values)
# shap.plots.bar(explainer.expected_value, shap_values)


# shap.force_plot(explainer.expected_value, shap_values, test_dataset)
# shap.summary_plot(shap_values,test_dataset)


# shap.summary_plot(shap_values,test_dataset,max_display=30)
# shap.decision_plot(explainer.expected_value,shap_values[:200])


# shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(test_dataset)
# shap.summary_plot(shap_interaction_values, test_dataset)

# shap.dependence_plot(
#     ("Age", "BMI"),
#     shap_interaction_values, X.iloc[:2000,:],
#     display_features=X_display.iloc[:2000,:]
# )

# import matplotlib.pylab as pl
# import numpy as np
# tmp = np.abs(shap_interaction_values).sum(0)
# for i in range(tmp.shape[0]):
#     tmp[i,i] = 0
# inds = np.argsort(-tmp.sum(0))[:50]
# tmp2 = tmp[inds,:][:,inds]
# pl.figure(figsize=(12,12))
# pl.imshow(tmp2)
# pl.yticks(range(tmp2.shape[0]), rotation=50.4, horizontalalignment="right")
# pl.xticks(range(tmp2.shape[0]), rotation=50.4, horizontalalignment="left")
# pl.gca().xaxis.tick_top()
# pl.show()