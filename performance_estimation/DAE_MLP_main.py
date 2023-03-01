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
from model import DAE_MLP, DACBIP_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
# %%
# setting up logger
logfile = "./log/DAE_MLP.log"
make_file_dir(logfile)
logger = logging.getLogger("DAE_MLP logger")
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
                        default='/home/yyx/interference_prediction/Alioth/Result/DAE_Pred0.csv', help="data directory")
    parser.add_argument("--data_dir2", type=str,
                        default='/home/yyx/interference_prediction/dacbip/cassandra_150_data.csv')
    parser.add_argument("--config", type=str,
                        default="/home/yyx/interference_prediction/dacbip/config/DAE_MLP_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
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

class DAE_MLP_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = DAE_MLP(args).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.save_model = args.save_model

        self.dataset = DACBIP_random_dataset(args.data_dir)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.test_dataset = DACBIP_random_dataset(args.data_dir)
        self.test_dataloader = DataLoader(self.test_dataset)
        self.t_dataset = DACBIP_random_dataset(args.data_dir2)
        # self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        # self.test_dataset = DACBIP_random_dataset(args.data_dir)
        self.t_dataloader = DataLoader(self.t_dataset,batch_size=args.batch_size, shuffle=True)

    def _train(self, epoch):
        self.model.train()
        for idx, (VM, SoI, labels) in enumerate(self.train_dataloader):
            VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(VM, SoI)
            output = torch.flatten(output)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            if idx % self.args.log_interval == 0:
                logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(VM), len(self.train_dataset), loss.item()
                ))

    def _test(self, epoch):
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for VM, SoI, labels in self.test_dataloader:
                VM, SoI, labels = VM.to(self.device), SoI.to(self.device), labels.to(self.device)
                output = self.model(VM, SoI)
                output = torch.flatten(output)
                loss = self.criterion(output, labels)
                test_loss += loss.item()
                OUTPUT.append(output.float())
                LABELS.append(labels.float())
        test_loss /= len(self.test_dataset)
        logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, OUTPUT, LABELS

    def _save_model_state_dict(self):
        save_filename = "/home/yyx/interference_prediction/dacbip/model/DAE_MLP3.pt"
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
        if self.save_model:
            self._save_model_state_dict()
        nni.report_final_result(test_loss)
        storFile(labels,'/home/yyx/DAE_MLP_True.csv')
        storFile(output,'/home/yyx/DAE_MLP_Pred.csv')

    def test_machine(self,args):
        test_model = DAE_MLP(args)
        test_model.load_state_dict(torch.load('/home/yyx/interference_prediction/dacbip/model/DAE_MLP3.pt'))
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for VM, SoI, labels in self.t_dataloader:
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

# %%
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True

def main(args):
    logger.debug("Create DAE_MLP trainer")
    dae_mlp_t = DAE_MLP_trainer(args)
    dae_mlp_t.trainer()

def main2(args):
    dae_mlp_t = DAE_MLP_trainer(args)
    OUTPUT, LABELS = dae_mlp_t.test_machine(args)
    print(OUTPUT.mean(), LABELS.mean())  

# %%
# Get and update args
setup_seed(30)
cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
print(cmd_args)
tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
args = merge_parameter(args, tuner_args)
logger.info(args)

# logger.debug("Create DAE_MLP trainer")
# dae_mlp_t = DAE_MLP_trainer(args)
# dae_mlp_t.trainer()
main2(args)
# main(args)#////////////////////////////////////////////