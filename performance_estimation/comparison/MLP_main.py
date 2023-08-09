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
from model import MLP, MLP_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
from sklearn import metrics
# %%
# setting up logger
logfile = "./log/MLP.log"
make_file_dir(logfile)
logger = logging.getLogger("MLP logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

#"/home/yyx/interference_prediction/Alioth/Result/xgb_affs_new0_40.npy"
#"/home/yyx/interference_prediction/Alioth/Result/xgb_affs_new0_80.npy"
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder")
    # parser.add_argument("--data_dir", type=str,
    #                     default="/home/yyx/DP/classified1/0_data.csv", help="data directory")####dir
    parser.add_argument("--data_dir", type=str,
                        default='/home/yyx/interference_prediction/Alioth/Result/DAE_Pred00.npy', help="data directory")   
    parser.add_argument("--data_dir2", type=str,
                        default='/home/yyx/DP/classified_total/0_data_NEW.csv')#####dir
    parser.add_argument("--config", type=str,
                        default="/home/yyx/interference_prediction/Alioth/config/DAE_MLP_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=1, metavar='N',
                        help='save model or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')

    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    # args - parser.parse_args()
    return args

class MLP_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = MLP(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model

        self.dataset = MLP_random_dataset(args.data_dir,args.data_dir2)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset)

    def _train(self, epoch):
        self.model.train()
        for idx, (data ,label) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            label = label.to(self.device)
            # print(data.shape)
            # print(label.shape)
            self.optimizer.zero_grad()
            output = self.model(data)
            # print("output", output)
            
            # print("label",label)
            # break
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(data), len(self.train_dataset), loss.item()
                ))
    
    def _test(self, epoch):
        OUTPUT = []
        LABELS= []
        self.model.eval()
        # mape = 0
        test_loss = 0.
        with torch.no_grad():
            for data, label in self.test_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                label = label.to(self.device)
                loss = self.criterion(output, label)
                test_loss += loss.item()
                OUTPUT.append(output.float())
                LABELS.append(label)            
                # test_mape = mape_loss_func(output, label)
                # mape += l1_loss_fn(output, label).item()
        test_loss /= len(self.test_dataset)
        # mape /= len(self.test_dataset)
        # print("label:", label) 
        # print("output:", output)  
        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, OUTPUT, LABELS

    def _save_model_state_dict(self,save_filename="/home/yyx/interference_prediction/Alioth/model/MLP0.pt"):
        # save_filename = #/home/yyx/interference_prediction/Alioth/config/DAE_MLP_config.json
        make_file_dir(save_filename)
        
        model_st = self.model.state_dict()
        k = list(model_st.keys())
        # for i in k:
        #     if "FC" not in i:
        #         del model_st[i]
        print(k)
        torch.save(model_st, save_filename)

    def trainer(self):
        for epoch in range(self.args.epochs):
            #print(self.args.epochs, epoch, "_-----------------------")
            self._train(epoch)
            test_loss, output, labels = self._test(epoch)
            nni.report_intermediate_result(test_loss)
        # if self.save_model:
        #     self._save_model_state_dict()

        nni.report_final_result(test_loss)
        print("MAE:",metrics.mean_absolute_error(labels, output))

        print(output)
        dirpre = '/home/yyx/interference_prediction/Alioth/Result/MLP_Pred0_affs.npy'
        dirlabel = '/home/yyx/interference_prediction/Alioth/Result/MLP_True0_affs.npy'

        np.save(dirlabel,np.array(labels))
        np.save(dirpre,np.array(output))
        return output, labels

    def test_machine(self,args):
        test_model = MLP(args)
        test_model.load_state_dict(torch.load('/home/yyx/interference_prediction/dacbip/model/MLP0.pt'))
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for VM, SoI, labels in self.test_dataloader:
                # VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
                pred = test_model(VM, SoI)
                pred = torch.flatten(pred)
                loss = self.criterion(pred, labels)
                test_loss += loss.item()
                #print(VM.shape)
                OUTPUT.append(torch.mean(pred.float()).item())
                LABELS.append(torch.mean(labels.float()).item())
        # test_loss /= len(t_dataset)
        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return  np.array(OUTPUT), np.array(LABELS)

    def save(self,save_filename,use_pt=True):
        if use_pt:
            self._save_model_state_dict(save_filename)
        else:
            # require no .pt in save_filename
            torch.save(self.model,save_filename)

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True

def MLP_main(args,):
    logger.debug("Create MLP trainer")
    dae_mlp_t = MLP_trainer(args)
    output, labels = dae_mlp_t.trainer()
    dae_mlp_t.save("/home/yyx/interference_prediction/Alioth/model/SHAP_DAE_MLP0_affs.pt")
    print(output)
    print(labels)
    # j=k=0#真阳性
    # m=n=0#假阳性
    # for i in range(len(output)):
    #     if output[i]< -0.05 and labels[i]< -0.05:
    #         j+=1
    #     if output[i]< -0.10 and labels[i]< -0.10:
    #         k+=1
    #     if output[i]< -0.05 and labels[i]> -0.05:
    #         m+=1
    #     if output[i]< -0.10 and labels[i]< -0.10:
    #         n+=1
    # print("Precision_0.05:",j/(j+m))
    # print("Precision_0.10:",k/(k+n))

def main2(args):
    dae_mlp_t = MLP_trainer(args)
    OUTPUT, LABELS = dae_mlp_t.test_machine(args)
    print(OUTPUT.mean(), LABELS.mean())  

# %%
# Get and update args
# setup_seed(30)


cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
# # print(cmd_args)

tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
args = merge_parameter(args, tuner_args)

logger.info(args)
logger.debug("Create MLP trainer")
MLP_main(args)
# # logger.debug("Create DAE_MLP trainer")
# # dae_mlp_t = DAE_MLP_trainer(args)
# # dae_mlp_t.trainer()
# # main2(args)
# MLP_main(args)#////////////////////////////////////////////