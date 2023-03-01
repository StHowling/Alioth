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
from model import LinearRegression, MLP_random_dataset
from utils import merge_parameter, load_args, make_file_dir, storFile
from sklearn.model_selection import train_test_split
from sklearn import metrics
# %%
# setting up logger
logfile = "./log/LRG.log"
make_file_dir(logfile)
logger = logging.getLogger("LRG logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder")
    # parser.add_argument("--data_dir", type=str,
    #                     default="/home/yyx/interference_prediction/Alioth/Result/DAE_Pred00_affs.npy", help="data directory")
    parser.add_argument("--data_dir", type=str,
                        default="/home/yyx/interference_prediction/Alioth/Result/DAE_Pred00.npy", help="data directory")                        
    parser.add_argument("--data_dir2", type=str,
                        default='/home/yyx/DP/classified_total/0_data_NEW.csv')#####dir                        
    parser.add_argument("--config", type=str,
                        default="/home/yyx/interference_prediction/Alioth/config/LRG_config.json", help="config filename")
    parser.add_argument('--save_model', type=int, default=0, metavar='N',
                        help='save model or not')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--cuda', type=int, default=1, metavar='N',
                        help='use CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(args=["--log_interval", "5", "--save_model", "1"])
    # args - parser.parse_args()
    return args


class LinearRegression_trainer():
    def __init__(self, args):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.criterion = nn.MSELoss()
        # print("args", args)
        self.model = LinearRegression(args).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.dataset = MLP_random_dataset(args.data_dir,args.data_dir2)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset,test_size=0.2,random_state=0)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset)


    def _train(self, epoch):
        self.model.train()
        for idx, (data ,label) in enumerate(self.train_dataloader):
            data = data.to(self.device)
            label = label.to(self.device)
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
        
    def trainer(self):
        for epoch in range(self.args.epochs):
            #print(self.args.epochs, epoch, "_-----------------------")
            self._train(epoch)
            test_loss, output, labels = self._test(epoch)
            nni.report_intermediate_result(test_loss)
            print("MAE:",metrics.mean_absolute_error(labels, output))
        nni.report_final_result(test_loss)
        print(output)
        j=k=0#真阳性
        m=n=0#假阳性
        for i in range(len(output)):
            if output[i]< -0.05 and labels[i]< -0.05:
                j+=1
            if output[i]< -0.10 and labels[i]< -0.10:
                k+=1
            if output[i]< -0.05 and labels[i]> -0.05:
                m+=1
            if output[i]< -0.10 and labels[i]< -0.10:
                n+=1
        print("Precision_0.05:",j/(j+m))
        print("Precision_0.10:",k/(k+n))
        # storFile(LABELS,'/home/yyx/DAE_LRG_True.csv')
        # storFile(OUTPUT,'/home/yyx/DAE_LRG_Pred.csv')     

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True


def main(args):
    logger.debug("Create LinearRegression trainer")
    LinearRegression_t = LinearRegression_trainer(args)
    LinearRegression_t.trainer()

# %%
# Get and update args
# setup_seed(20)
cmd_args = EasyDict(vars(get_parameters()))
args = load_args(cmd_args.config)
tuner_args = nni.get_next_parameter()
args = merge_parameter(args, cmd_args)
args = merge_parameter(args, tuner_args)

logger.info(args)

main(args)