# %%
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from  torch.nn import init
from utils import load_args, save_json
from torch.autograd import Variable
from easydict import EasyDict
from typing import Dict, List
# %%
class DAE(nn.Module):
    def __init__(self, args):
        super(DAE, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(args.input_size, args.encoder_sizes[0]))
        self.encoder.append(nn.ReLU())
        for i in range(len(args.encoder_sizes) - 1):
            self.encoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i + 1]))
            self.encoder.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for i in range(len(args.decoder_sizes) - 1, 0, -1):
            self.decoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i - 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(args.encoder_sizes[0], args.output_size))
        self.decoder.append(nn.ReLU())
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
        return x

class pretrained_encoder(nn.Module):
    def __init__(self, args):
        if not isinstance(args,EasyDict):
            assert isinstance(args,Dict), "args must be a dict or EasyDict"
            args = EasyDict(args)
        super(pretrained_encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(args.input_size, args.encoder_sizes[0]))
        self.encoder.append(nn.ReLU())
        for i in range(len(args.encoder_sizes) - 1):
            self.encoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i + 1]))
            self.encoder.append(nn.ReLU())

        
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
        return x
    
    def load_state_dict_from_DAE(self, filename):
        '''
        Partially load the state dictionary from a DAE, only using params of its encoder
        '''
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'encoder' in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Loaded pretrained encoder from {}'.format(filename))

def load_encoder_state_dict(pretrained_encoder, filename):
    pretrained_encoder.load_state_dict(torch.load(filename), strict=False)

def load_dae_state_dict(DAE, filename):
    DAE.load_state_dict(torch.load(filename), strict=False)



class DAE_MLP(nn.Module):
    def __init__(self, args):
        super(DAE_MLP, self).__init__()
        self.VM_encoder = pretrained_encoder(args, "VM_")
        load_encoder_state_dict(self.VM_encoder, args.VM_encoder_model)
        self.SoI_encoder = pretrained_encoder(args, "SoI_")
        load_encoder_state_dict(self.SoI_encoder, args.SoI_encoder_model)

        self.fc_input_size = args.VM_hidden_size + args.SoI_hidden_size
        # print("self.fc_input_size", self.fc_input_size)
        self.FC = nn.ModuleList()
        self.FC.append(nn.Linear(self.fc_input_size, args.fc_sizes[0]))
        self.FC.append(nn.ReLU())
        for i in range(len(args.fc_sizes) - 1):
            self.FC.append(nn.Linear(args.fc_sizes[i], args.fc_sizes[i + 1]))
            self.FC.append(nn.ReLU())
        self.FC.append(nn.Linear(args.fc_sizes[-1], args.output_size))
        self.FC.append(nn.ReLU())
        
    def forward(self, VM, SoI):
        VM_features = self.VM_encoder(VM)
        SoI_features = self.SoI_encoder(SoI)
        # print(x.shape)
        x = torch.cat((VM_features, SoI_features), 1)
        for idx, layer in enumerate(self.FC):
            x = layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP,self).__init__()
        self.fc_input_size = 40
        # print("self.fc_input_size", self.fc_input_size)
        self.FC = nn.ModuleList()
        self.FC.append(nn.Linear(self.fc_input_size, args.fc_sizes[0]))
        self.FC.append(nn.ReLU())
        for i in range(len(args.fc_sizes) - 1):
            self.FC.append(nn.Linear(args.fc_sizes[i], args.fc_sizes[i + 1]))
            self.FC.append(nn.ReLU())
        self.FC.append(nn.Linear(args.fc_sizes[-1], args.output_size))
        self.FC.append(nn.ReLU())
    def forward(self, x):
        # x = torch.cat((VM_features, SoI_features), 1)
        for idx, layer in enumerate(self.FC):
            x = layer(x)
        return x


class LinearRegression(nn.Module):
    def __init__(self, args):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(40,1)#输入维度248 = 219+29

    def forward(self, x):
        # x = torch.cat((VM_features, SoI_features), 1)
        out = self.linear(x)
        return out

class LRG_random_dataset(Dataset):
    def __init__(self, csv_file, csv_file2):
        self.raw_data = np.load(csv_file,allow_pickle=True)
        # self.raw_data = pd.read_csv(csv_file)
        self.raw_data = torch.from_numpy(self.raw_data.astype(float))
        self.raw_data = torch.tensor(self.raw_data, dtype=torch.float32)
        self.label_data = pd.read_csv(csv_file2).drop(['workload_intensity'], axis=1)
        self.label_data = torch.from_numpy(self.label_data.values).float()
        self.label = self.label_data[:,-1] + 1

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx], self.label[idx]

class MLP_random_dataset(Dataset):
    def __init__(self, csv_file, csv_file2):
        self.raw_data = np.load(csv_file,allow_pickle=True)
        self.raw_data = torch.from_numpy(self.raw_data.astype(float))
        self.raw_data = torch.tensor(self.raw_data, dtype=torch.float32)
        self.label_data = pd.read_csv(csv_file2).drop(['workload_intensity'], axis=1)
        self.label_data = torch.from_numpy(self.label_data.values).float()
        # self.raw_data = torch.from_numpy(self.raw_data).float()
        # self.raw_data = torch.from_numpy(self.raw_data.astype(float))
        # self.raw_data = np.squeeze(self.raw_data)
        # self.x = torch.cat((self.raw_data), 1)       
        self.label = self.label_data[:,-1]
        # self.raw_data = self.label_data[:,:219]
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        return self.raw_data[idx], self.label[idx]



class CART_random_dataset(Dataset):
    def __init__(self, csv_file):
        self.raw_data = pd.read_csv(csv_file)
        self.raw_data = np.array(self.raw_data.values, dtype = 'float32')
        self.LRG_raw_data1 = self.raw_data[:,0:219]
        self.LRG_raw_data2 = self.raw_data[:,439:468] 
        self.x = np.concatenate((self.LRG_raw_data1,self.LRG_raw_data2), 1)     
        self.label = self.raw_data[:,-1] + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


    # '0':['hbase','cassandra','mongoDB'],
    # '1':['kafka','rabbitmq'],
    # '2':['etcd','redis'],
    # '3':['milc'],

class DAE_random_dataset(Dataset):
    def __init__(self, csv_file, selected_apps:List = None,  normalize='minmax'):
        raw_data = pd.read_csv(csv_file)
        if selected_apps is not None:
            raw_data = raw_data.loc[raw_data['app'].isin(selected_apps),:].reset_index(drop=True)
        if normalize == 'minmax':
            scaler = MinMaxScaler()
        elif normalize == 'standard':
            scaler = StandardScaler()
        else:
            raise NotImplementedError
        metric_vals = raw_data.iloc[:,:-5]
        array_to_fit = np.concatenate([metric_vals.values[:, :metric_vals.shape[1]//2], metric_vals.iloc[:, metric_vals.shape[1]//2:].drop_duplicates().values],axis=0)
        scaler.fit(array_to_fit)

        self.stressed_metrics = torch.Tensor(scaler.transform(metric_vals.values[:, :metric_vals.shape[1]//2]))
        self.label = torch.Tensor(scaler.transform(metric_vals.values[:, metric_vals.shape[1]//2:]))
        

    def __len__(self):
        return len(self.stressed_metrics)

    def __getitem__(self, idx):
        return self.stressed_metrics[idx], self.label[idx]
 


# %%
# Test DAE
# args = load_args("/home/yyx/interference_prediction/dacbip/config/DAE_config.json")
# a = DAE(args)
# data = pd.read_csv('/home/yyx/interference_prediction/dacbip/0_data.csv')
# data = torch.from_numpy(data.values).float()
# print(data[:,467])
#dae_dataset = DAE_random_dataset('/home/yyx/interference_prediction/dacbip/cassandra-all_merged.csv')
# dae_dataset = DAE_random_dataset(100,args)
#dae_dataloader = DataLoader(dae_dataset, batch_size=args.batch_size, shuffle=True)
#print(len(dae_dataset))
# for data in dae_dataloader:
#     print(data.shape)
#     break
# %%
# Test pretrained_encoder
# args = load_args("config/DACBIP_config.json")
# VM_encoder = pretrained_encoder(args, "VM_")
# load_encoder_state_dict(VM_encoder, args.VM_encoder_model)


# %%
# Test DAE_MLP
# args = load_args("config/DAE_MLP_config.json")
# model = DAE_MLP(args)
# print(model)
# dataset = DACBIP_random_dataset(1000, args)
# dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
# for idx, (VM, SoI, labels) in enumerate(dataloader):
#     print(VM.shape, SoI.shape)
#     output = model(VM, SoI)
#     print(output.shape)
#     break

