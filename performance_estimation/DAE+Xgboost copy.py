# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
import xgboost
from new_data_utils import get_data
from model import DAE_new
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# %%
data_dir = "../data/output"
model_save_path = "model/Alioth"
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
# %%
# get and transform data without nostress features
new_df = pd.read_csv(os.path.join(data_dir, 'mul_all_intall_warming3_nosliding_all.csv'))
new_df.drop(new_df[new_df['app']=='noapp'].index,inplace=True)
new_df_ltc = get_data(new_df, "all", "latency")

numeric_cols = list(new_df_ltc.columns[:-5])
data = new_df_ltc.loc[(new_df_ltc['app']!=''), numeric_cols]
train_part, test_part = train_test_split(data, test_size=0.2, random_state=42)

transformations = [("scaler", MinMaxScaler(clip=False), list(train_part.columns)[:-1])]
preprocessor = ColumnTransformer(transformations,remainder="passthrough")

preprocessor.fit(train_part)
train_data = preprocessor.transform(train_part)
test_data = preprocessor.transform(test_part)
train_feature = train_data[:,:-1]
train_label = train_data[:,-1]
test_feature = test_data[:,:-1]
test_label = test_data[:,-1]

# %%
# load pretrained dae parameters
encoder = DAE_new(1090)
dae_paras = torch.load(os.path.join(model_save_path, "DAE.pt"))
# dae_paras = {k.replace("encoder.", ""): v for k, v in dae_paras.items() if k.replace("encoder.", "") in encoder.state_dict()}
encoder.load_state_dict(dae_paras)
encoder.to(device)
encoder.eval()
# %%
# use dae encoder to get dae processed features
train_feature_tensor = torch.Tensor(train_feature)
train_dataloader = DataLoader(train_feature_tensor, batch_size=32)
test_feature_tensor = torch.Tensor(test_feature)
test_dataloader = DataLoader(test_feature_tensor, batch_size=32)

train_dae_outputs = []
test_dae_outputs = []
for i in train_dataloader:
    i = i.to(device)
    output = encoder(i)
    train_dae_outputs.append(output)

for i in test_dataloader:
    i = i.to(device)
    output = encoder(i)
    test_dae_outputs.append(output)
# %%
train_dae_outputs = torch.cat(train_dae_outputs).detach().cpu().numpy()
test_dae_outputs = torch.cat(test_dae_outputs).detach().cpu().numpy()
train_feature = np.concatenate((train_feature, train_dae_outputs), axis=1)
test_feature = np.concatenate((test_feature, test_dae_outputs), axis=1)
print(train_feature.shape, test_feature.shape)

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
model = xgb.XGBRegressor( eta=0.01, colsample_bylevel=0.8, colsample_bytree = 0.6, learning_rate=0.03, max_depth=6, n_estimators=1000).fit(train_feature, train_label)
predict = model.predict(test_feature)
_ = xgb.plot_importance(model,height = 0.9)
FI = model.feature_importances_
# %%
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

for i in tqdm(range(len(predict))):
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

# print("是否发生中等劣化 Precision:", tp_mid/(tp_mid+fp_mid))
# print("是否是轻微劣化 Precision:", tp_min/(tp_min+fp_min))
# print("是否发生中等劣化 Recall:", tp_mid/(tp_mid+fn_mid))
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