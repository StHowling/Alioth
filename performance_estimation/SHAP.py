# %%
import os
from sklearn.model_selection import train_test_split
# import nni
import csv
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from easydict import EasyDict
from model import MLP, MLP_random_dataset, DAE, DAE_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile

from MLP_main import get_parameters
import shap
import time
shap.initjs()
from utils import storFile


cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
device = torch.device("cuda")

def transform(dataset):
    dt = []
    for item in dataset:
        dt.append(item[0])
    return np.array(dt,dtype="float")

def load_model(args,path):
    model = MLP(args)
    model.load_state_dict(torch.load(path))
    model.eval()
    
    # model = model.to(device)
    dataset = MLP_random_dataset(args.data_dir,args.data_dir2)
    X_train, X_test = train_test_split(dataset,test_size=0.2,random_state=0)
    X_train = transform(X_train)
    X_test = transform(X_test)

    return model, X_train, X_test

MLP_model, X_train, X_test = load_model(args,'/home/yyx/interference_prediction/Alioth/model/SHAP_test0.pt') 

def f(z):
    data_loader = DataLoader(z)
    pred = []
    with torch.no_grad():
        for data in data_loader:
            # data = data.to(device)
            output = MLP_model(data.float())
            pred.append(output.float())
    return np.array(pred)


# explainer = shap.KernelExplainer(f, X_train)
explainer = shap.SamplingExplainer(f, X_train)
shap_values = explainer.shap_values(X_test,n_samples=500)
shapvalue_sum = shap_values.sum(axis=0)
print(shapvalue_sum.shape)
# print(len(shap_values))
np.save("/home/yyx/interference_prediction/Alioth/Result/shap_values.npy",shap_values)
np.save("/home/yyx/interference_prediction/Alioth/Result/shapvalue_sum.npy",shapvalue_sum)
global_shap_values = pd.DataFrame(np.abs(shap_values).mean(0)).reset_index()
global_shap_values.columns = ["var","feature_importance"]
# # global_shap_values = global_shap_values.sort_values(feature_importance_, ascending = False)
# global_shap_values = global_shap_values.sort_values(feature_importance)
np.save("/home/yyx/interference_prediction/Alioth/Result/global_shap_values.npy",global_shap_values)
#%%
shap.force_plot(explainer.expected_value, shap_values, X_test)
shap.summary_plot(shap_values,X_test,plot_type="bar")
# shap.summary_plot(shap_values,X_test,max_display=30)
shap.decision_plot(explainer.expected_value,shap_values[:20])

# %%
# shap_interaction_values = explainer.shap_interaction_values(X_test,n_samples=500)
# storFile(shap_interaction_values,"/home/yyx/interference_prediction/Alioth/Result/shap_interaction_values.npy")
# shap.summary_plot(shap_interaction_values,X_test)
# shap.plots.force(shap_values)

# #%%
# shap.dependence_plot("RM",shap_values, X)
