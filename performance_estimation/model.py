# %%
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from  torch.nn import init
from utils import load_args, save_json
from torch.autograd import Variable
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
        # self.encoder.append(nn.Linear(args.encoder_sizes[-1], args.hidden_size))
        # self.encoder.append(nn.ReLU())
        self.decoder = nn.ModuleList()
        # self.decoder.append(nn.Linear(args.hidden_size, args.decoder_sizes[0]))
        # self.decoder.append(nn.ReLU())
        for i in range(len(args.decoder_sizes) - 1):
            self.decoder.append(nn.Linear(args.decoder_sizes[i], args.decoder_sizes[i + 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(args.decoder_sizes[-1], args.output_size))
        self.decoder.append(nn.ReLU())
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            # print(layer)
            # print("X_ENCODER:",x.shape) 
            x = layer(x)
        for idx, layer in enumerate(self.decoder):
            # print("X_DECODER:",x.shape) 
            x = layer(x)
        return x

class pretrained_encoder(nn.Module):
    def __init__(self, args, encoder_type):
        super(pretrained_encoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Linear(args[encoder_type + "input_size"], args[encoder_type + "encoder_sizes"][0]))
        self.encoder.append(nn.ReLU())
        for i in range(len(args[encoder_type + "encoder_sizes"]) - 1):
            self.encoder.append(nn.Linear(args[encoder_type + "encoder_sizes"][i], args[encoder_type + "encoder_sizes"][i + 1]))
            self.encoder.append(nn.ReLU())
        # self.encoder.append(nn.Linear(args[encoder_type + "encoder_sizes"][-1], args[encoder_type + "hidden_size"]))
        # self.encoder.append(nn.ReLU())
        
    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
        return x

def load_encoder_state_dict(pretrained_encoder, filename):
    pretrained_encoder.load_state_dict(torch.load(filename), strict=False)

def load_dae_state_dict(DAE, filename):
    DAE.load_state_dict(torch.load(filename), strict=False)


class DACBIP(nn.Module):
    def __init__(self, args):
        super(DACBIP, self).__init__()
        self.CNN = nn.ModuleList()
        self.CNN.append(nn.Conv2d(1, args.CNN_channels[0], kernel_size=args.CNN_kernel_size, padding=1))
        self.CNN.append(nn.ReLU())
        for i in range(len(args.CNN_channels) - 1):
            if args.CNN_channels[i + 1] == 0:
                self.CNN.append(nn.MaxPool2d(2,2))
            elif args.CNN_channels[i] == 0:
                self.CNN.append(nn.Conv2d(args.CNN_channels[i - 1], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
                self.CNN.append(nn.ReLU())
            else:
                self.CNN.append(nn.Conv2d(args.CNN_channels[i], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
                self.CNN.append(nn.ReLU())
        
        self.FC = nn.ModuleList()
        self.FC.append(nn.Linear(220, 32))###################
        self.FC.append(nn.ReLU())
        self.FC.append(nn.Linear(32, args.output_size))
        self.FC.append(nn.ReLU())

    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = torch.unsqueeze(x, dim=2)
        for idx, layer in enumerate(self.CNN):
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        # print("x:",x.shape)
        for idx, layer in enumerate(self.FC):
            x = layer(x)
        return x

class LSTM(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lstmLayer=nn.LSTM(args.in_dim,args.hidden_dim,args.layer_num)
        self.relu=nn.ReLU()
        self.fcLayer=nn.Linear(args.hidden_dim,args.out_dim)

        self.weightInit=(np.sqrt(1.0/args.hidden_dim))

    def forward(self, x):
        # print(x)
        # x = Variable(x)
        out,_=self.lstmLayer(x)######
        s,b,h=out.size() #seq,batch,hidden
        #out=out.view(s*b,h)
        out=self.relu(out)
        out=out[12:]
        out=self.fcLayer(out)
        #out=out.view(s,b,-1)
        return out

    #初始化权重
    def weightInit(self, gain=1):
            # 使用初始化模型参数
        for name, param in self.named_parameters():
            if 'lstmLayer.weight' in name:
                init.orthogonal(param, gain)


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

class LSTM_random_dataset(Dataset):
    def __init__(self, csv_file ,csv_file2):
        self.raw_data = np.load(csv_file,allow_pickle=True)
        # self.raw_data = pd.read_csv(csv_file)
        self.label_data = pd.read_csv(csv_file2).drop(['workload_intensity'], axis=1)
        self.label_data = torch.from_numpy(self.label_data.values).float()
        # self.raw_data = torch.from_numpy(self.raw_data).float()
        # self.raw_data = torch.from_numpy(self.raw_data.astype(float))
        # self.raw_data = np.squeeze(self.raw_data)
        # self.x = torch.cat((self.raw_data), 1)       
        self.label = self.label_data[:,-1] + 1
        # self.raw_data = self.raw_data.reshape(-1,1,1)

        # print(self.raw_data.shape)
        # print(type(self.raw_data))
        # print(self.label.shape)
        # print(type(self.label_data))
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
    def __init__(self, csv_file,app):
        self.raw_data = pd.read_csv(csv_file)
        self.raw_data = self.raw_data[(self.raw_data['APP'] == app )]
        self.raw_label = self.raw_data[(self.raw_data['stress_type'] == 'NO_STRESS' )]

        print("--------------------------------data-info---------------------------------------")

        print("workload_intensity,:",list(self.raw_label['WORKLOAD_INTENSITY'].unique()))

        # self.data = pd.DataFrame(data=None, columns = self.raw_data.columns, index = )
        self.data = pd.DataFrame(data=None)

        for i in list(self.raw_label['WORKLOAD_INTENSITY'].unique()):
            self.df = self.raw_data[(self.raw_data['WORKLOAD_INTENSITY'] == i )]
            self.metrics_nostress = self.df[self.df['stress_type']=='NO_STRESS'].drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP'], axis=1).mean().to_frame().T
            self.keys = list(self.metrics_nostress.columns)
            print(self.metrics_nostress)
            self.metrics_nostress = self.metrics_nostress.iloc[0, :self.keys.index("IPC")+1].to_frame().T
            for ik in self.metrics_nostress:#value_keys:
                #print(ik)
                self.df['Label_'+ik] = self.metrics_nostress[ik][0]
            self.data = pd.concat([self.data,self.df],axis = 0)

        # self.data.to_csv("/home/yyx/interference_prediction/Alioth/DATA/etcd+_.csv")
        self.data = self.data.drop(['WORKLOAD_INTENSITY','stress_intensity','CATERGORY','stress_type','APP','QoS','Unnamed: 0','Label_Unnamed: 0'], axis=1)
        print(self.data)
        self.data = np.array(self.data, dtype = 'float32')
        # self.data = torch.from_numpy(self.data.values).float()
        self.VM_raw_data = self.data[:,0:218]
        print(self.VM_raw_data)
        self.label = self.data[:,218:436]
        print(self.label)

    def __len__(self):
        return len(self.VM_raw_data)

    def __getitem__(self, idx):
        return self.VM_raw_data[idx], self.label[idx]
 

# class DAE_random_dataset(Dataset):
#     def __init__(self, size, args):
#         self.raw_data = torch.rand(size, args.input_size)
#     def __len__(self):
#         return len(self.raw_data)

#     def __getitem__(self, idx):
#         return self.raw_data[idx]

class DACBIP_random_dataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = torch.from_numpy(self.data.values).float()
        self.VM_raw_data = self.data[:,0:219]
        self.SoI_raw_data = self.data[:,439:468]
        self.label = self.data[:,-1] + 1
        #self.label = torch.from_numpy(self.label.values).float()
        #print(self.label.shape, self.VM_raw_data[0])
    
    def __len__(self):
        return len(self.VM_raw_data)
    
    def __getitem__(self, idx):
        return self.VM_raw_data[idx], self.SoI_raw_data[idx], self.label[idx]

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
# Test DACBIP
# args = load_args("config/DACBIP_config.json")
# model = DACBIP(args)
# print(model)
# dataset = DACBIP_random_dataset(1000, args)
# dataloader = DataLoader(dataset, args.batch_size, shuffle=True)
# for idx, (VM, SoI, labels) in enumerate(dataloader):
#     print(VM.shape, SoI.shape)
#     output = model(VM, SoI)
#     print(output.shape)
#     break
# s = model.state_dict()
# print(s.keys())
# k = list(s.keys())
# for i in k:
#     if "CNN" not in i and "FC" not in i:
#         del s[i]
# print(s.keys())

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
# %%
# a = torch.rand(32, 16)
# b = torch.rand(32, 16)
# c = torch.cat((a, b), 1)
# print(c.shape)
# a = torch.unsqueeze(a, 2)
# b = torch.unsqueeze(b, 1)
# print(a.shape, b.shape)
# c = torch.matmul(a, b)
# print(c.shape)

#%%

# class DACBIP(nn.Module):
#     def __init__(self, args):
#         super(DACBIP, self).__init__()
#         self.VM_encoder = DAE(args, "VM_")
#         load_dae_state_dict(self.VM_encoder, args.VM_encoder_model)
#         # self.SoI_encoder = pretrained_encoder(args, "SoI_")
#         # load_encoder_state_dict(self.SoI_encoder, args.SoI_encoder_model)
#         self.CNN = nn.ModuleList()
#         self.CNN.append(nn.Conv2d(1, args.CNN_channels[0], kernel_size=args.CNN_kernel_size, padding=1))
#         self.CNN.append(nn.ReLU())
#         for i in range(len(args.CNN_channels) - 1):
#             if args.CNN_channels[i + 1] == 0:
#                 self.CNN.append(nn.MaxPool2d(2, 2))
#             elif args.CNN_channels[i] == 0:
#                 self.CNN.append(nn.Conv2d(args.CNN_channels[i - 1], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
#                 self.CNN.append(nn.ReLU())
#             else:
#                 self.CNN.append(nn.Conv2d(args.CNN_channels[i], args.CNN_channels[i + 1], kernel_size=args.CNN_kernel_size, padding=1))
#                 self.CNN.append(nn.ReLU())
        
#         self.FC = nn.ModuleList()
#         self.FC.append(nn.Linear(16, 8))##!
#         self.FC.append(nn.ReLU())
#         self.FC.append(nn.Linear(8, args.output_size))
#         self.FC.append(nn.ReLU()
#     def forward(self, VM, SoI):
#         VM_features = self.VM_encoder(VM)
#         SoI_features = self.SoI_encoder(SoI)
#         # print("VM",VM_features)
#         # print("SOI",SoI_features)
#         # VM_features = torch.unsqueeze(VM_features, 2)
#         # SoI_features = torch.unsqueeze(SoI_features, 1)
#         # print(VM_features.shape)
#         # print(SoI_features.shape)
#         x = torch.cat((VM_features, SoI_features),1)
#         # x = torch.mul(VM_features, SoI_features)
#         # print("x:", x)
#         x = torch.unsqueeze(x, 1)
#         # print("x:", x)
#         # for idx, layer in enumerate(self.CNN):
#         #     x = layer(x)
#             # print(x.shape)
#             # print(layer)
#         x = torch.flatten(x, start_dim=1)
#         # print("x:", x)
#         # print(x.shape)
#         for idx, layer in enumerate(self.FC):
#             x = layer(x)
#         # print("X:", x)
#         return x


class DADAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, VM, SoI):
        pass