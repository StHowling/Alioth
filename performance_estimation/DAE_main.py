# %%
import os
import nni
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
from model import DAE, DAE_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split


# %%
# setting up logger
logfile = "./log/DAE_VM.log"
# logfile = "./log/DAE_VM.log"
make_file_dir(logfile)
logger = logging.getLogger("DAE logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder")
    parser.add_argument("--data_dir", type=str,
                        default="/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv", help="data directory")
    parser.add_argument("--app1", type=str,
                        default='cassandra')
    parser.add_argument("--app2", type=str,
                        default='rabbitmq')
    parser.add_argument("--config", type=str,
                        default="/home/yyx/interference_prediction/Alioth/config/DAE_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 32)')
    # parser.add_argument("--hidden_size", type=int, default=16, metavar='N',
    #                     help='hidden layer size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(args=["--data_dir", "/home/yyx/interference_prediction/Alioth/DATA/all_merged_by_app_095.csv", "--log_interval", "5", "--save_model", "1"])#####dir
    # args - parser.parse_args()
    return args 

class DAE_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = DAE(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model

        self.dataset = DAE_random_dataset(args.data_dir,args.app1)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2, random_state=1)
        self.train_dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True) 
        self.test_dataloader = DataLoader(self.dataset)
        self.t_dataset = DAE_random_dataset(args.data_dir,args.app2)
        self.t_dataloader = DataLoader(self.t_dataset,batch_size=args.batch_size, shuffle=True)

    def _train(self, epoch):
        self.model.train()
        for idx, (data ,label) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            # print(output)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(data), len(self.dataset), loss.item()
                ))
    
    def _test(self, epoch):
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for data, label in self.test_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                label = label.to(self.device)
                loss = self.criterion(output, label)
                test_loss += loss.item()
                OUTPUT.append(output.float().cpu().numpy()[0])
                LABELS.append(label.cpu().numpy()[0])

        test_loss /= len(self.dataset)

        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, OUTPUT, LABELS
    
    def _save_model_state_dict(self):
        save_filename = "/home/yyx/interference_prediction/Alioth/model/DAE_cassandra.pt"
        make_file_dir(save_filename)
        torch.save(self.model.state_dict(), save_filename)

    def trainer(self):
        dirpre = '/home/yyx/interference_prediction/Alioth/Result/DAE_Predcassandra.npy'
        dirlabel = '/home/yyx/interference_prediction/Alioth/Result/DAE_Truecassandra.npy'

        for epoch in range(self.args.epochs):
            self._train(epoch)
            test_loss, OUTPUT, LABELS = self._test(epoch)
            nni.report_intermediate_result(test_loss)
        if self.save_model:
            self._save_model_state_dict()
        nni.report_final_result(test_loss)
        # print(np.array(LABELS).shape)
        #print(LABELS.shape())   

        # np.save(dirlabel,np.array(LABELS))
        # np.save(dirpre,np.array(OUTPUT))
        return dirpre

    def test_machine(self,args):
        test_model = DAE(args)
        test_model.load_state_dict(torch.load('/home/yyx/interference_prediction/Alioth/model/DAE_cassandra.pt'))
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.

        with torch.no_grad():
            for data, label in self.t_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                label = label.to(self.device)
                loss = self.criterion(output, label)
                test_loss += loss.item()
                OUTPUT.append(output.float().cpu().numpy()[0])
                LABELS.append(label.cpu().numpy()[0])

        # test_loss /= len(self.dataset)

        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, OUTPUT, LABELS

def DAE_main(args):
    dae_t = DAE_trainer(args)
    dirpre = dae_t.trainer()
    return dirpre

def main2(args):
    dae_t = DAE_trainer(args)
    test_loss, OUTPUT, LABELS = dae_t.test_machine(args)
    print(np.mean(OUTPUT), np.mean(LABELS))  

#/home/yyx/interference_prediction/Alioth/config
# %%
# Get and update args
l1_loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
cmd_args = EasyDict(vars(get_parameters()))
#args = load_args(cmd_args.config)
args = load_args("/home/yyx/interference_prediction/Alioth/config/DAE_config.json")
tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
args = merge_parameter(args, tuner_args)
logger.info(args)
logger.debug("Create DAE trainer")
# logger.debug("Create DAE trainer")
# dae_t = DAE_trainer(args)
# dae_t.trainer()
main2(args)
# DAE_main(args)
# %%
# Test code
# for i in dae_t.model.state_dict():#输出参数
#     print(i, dae_t.model.state_dict()[i].size())
# for i in dae_t.optimizer.state_dict():
#     print(i, dae_t.optimizer.state_dict()[i])
# torch.save(dae_t.model.state_dict(), "DAE_VM.pt")#保存模型参数
#torch.save(dae_t.model.state_dict(), "DAE_SOI.pt")#保存模型参数