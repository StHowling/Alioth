'''
Adpated from Jindong Wang's famous transferlearning repo on Github:
https://github.com/jindongwang/transferlearning/blob/master/code/DeepDA
'''
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from model import pretrained_encoder
from sklearn.model_selection import train_test_split
from typing import List
from utils import *

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class AdversarialLoss(nn.Module):
    '''
    Acknowledgement: The adversarial loss implementation is inspired by http://transfer.thuml.ai/
    '''
    def __init__(self, gamma=1.0, max_iter=1000, use_lambda_scheduler=True, **kwargs):
        super(AdversarialLoss, self).__init__()
        self.domain_classifier = Discriminator(**kwargs)
        self.use_lambda_scheduler = use_lambda_scheduler
        if self.use_lambda_scheduler:
            self.lambda_scheduler = LambdaSheduler(gamma, max_iter)
        
    def forward(self, source, target):
        lamb = 1.0
        if self.use_lambda_scheduler:
            lamb = self.lambda_scheduler.lamb()
            self.lambda_scheduler.step()
        source_loss = self.get_adversarial_result(source, True, lamb)
        target_loss = self.get_adversarial_result(target, False, lamb)
        adv_loss = 0.5 * (source_loss + target_loss)
        return adv_loss
    
    def get_adversarial_result(self, x, source=True, lamb=1.0):
        x = ReverseLayerF.apply(x, lamb)
        domain_pred = self.domain_classifier(x)
        device = domain_pred.device
        if source:
            domain_label = torch.ones(len(x), 1).long()
        else:
            domain_label = torch.zeros(len(x), 1).long()
        loss_fn = nn.BCELoss()
        loss_adv = loss_fn(domain_pred, domain_label.float().to(device))
        return loss_adv
    

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    

class Decoder(nn.Module):
    def __init__(self, decoder_sizes, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(decoder_sizes[-1], output_size))
        self.decoder.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x   
    
    def load_state_dict_from_DAE(self, filename):
        '''
        Partially load the state dictionary from a DAE, only using params of its deocder
        '''
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'decoder' in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Loaded pretrained decoder from {}'.format(filename))


class ReconstructionLoss(nn.Module):
    def __init__(self, decoder_sizes, output_size) -> None:
        super(ReconstructionLoss,self).__init__()
        self.decoder_sizes = decoder_sizes
        self.decoder = nn.ModuleList()
        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(decoder_sizes[-1], output_size))
        self.decoder.append(nn.ReLU())

    def forward(self, x, y):
        for layer in self.decoder:
            x = layer(x)
        loss = F.mse_loss(x, y)
        return loss
    
    def load_state_dict_from_DAE(self, filename):
        '''
        Partially load the state dictionary from a DAE, only using params of its deocder
        '''
        pretrained_dict = torch.load(filename)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'decoder' in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Loaded pretrained decoder from {}'.format(filename))


class DADAE_random_dataset(Dataset):
    def __init__(self, csv_file, apps_to_leave_out:List, target_col='qos1', type='dae'):
        self.raw_data = pd.read_csv(csv_file)
        self.type = type
        idx = self.raw_data['app']!= apps_to_leave_out[0]
        for i in apps_to_leave_out[1:]:
            idx = idx & (self.raw_data['app']!=apps_to_leave_out[i])
        if self.type == 'dae':   
            self.raw_data = self.raw_data[idx]
            apps = set(self.raw_data['app'].unique()) - set(apps_to_leave_out)
        else:
            self.raw_data = self.raw_data[~idx] 
            apps = apps_to_leave_out

        self.raw_label = self.raw_data[(self.raw_data['stress_type'] == 'NO_STRESS' )]
        self.data = pd.DataFrame(data=None)
        
        for app in apps:
            for i in list(self.raw_label['workload'].unique()):
                self.df = self.raw_data.loc[(self.raw_data['workload'] == i ) & (self.raw_data['app'] == app)]
                self.metrics_nostress = self.df[self.df['stress_type']=='NO_STRESS'].drop(['workload','stress_intensity','stress_type','app', 'interval'], axis=1).mean().to_frame().T
                self.keys = list(self.metrics_nostress.columns)
                # print(self.metrics_nostress)
                self.metrics_nostress = self.metrics_nostress.iloc[0, :self.keys.index(target_col)].to_frame().T
                for ik in self.metrics_nostress:#value_keys:
                    #print(ik)
                    self.df['Label_'+ik] = self.metrics_nostress[ik][0]
                self.data = pd.concat([self.data,self.df],axis = 0)

        self.data = self.data.drop(['timestamp','workload','stress_intensity','stress_type','app','qos1', 'qos2', 'interval','Unnamed: 0','Label_Unnamed: 0'], axis=1).values.astype(np.float32)
        self.VM_raw_data = self.data[:, :self.data.shape[1]//2]
        self.label = self.data[:, self.data.shape[1]//2:]
            

    def __len__(self):
        return len(self.VM_raw_data)

    def __getitem__(self, idx):    
        return self.VM_raw_data[idx], self.label[idx]



class DADAE_trainer(object):
    def __init__(self, args, logger=None):
        self.args = args
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if logger is not None:
            self.logger = logger
        else:
            self.logger = self._get_default_logger()
    
        self.encoder = pretrained_encoder(args)
        self.decoder = Decoder(args.decoder_sizes, args.input_dim)
        self.encoder.load_state_dict_from_DAE(args.dae_path)
        self.decoder.load_state_dict_from_DAE(args.dae_path)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.reconstruction_criterion = nn.MSELoss()
        self.adaptation_criterion = AdversarialLoss().to(self.device)
        #!
        initial_lr = args.lr if not args.lr_scheduler else 1.0
        self.optimizer = optim.Adam(self._get_parameters(initial_lr=initial_lr), lr=args.lr, weight_decay=args.weight_decay)
        self.save_model = args.save_model

        self.dae_dataset = DADAE_random_dataset(args.data_dir,args.apps_to_leave_out, type='dae')
        self.dae_dataloader = DataLoader(self.dae_dataset, batch_size=args.batch_size_dae, shuffle=True, num_workers=4)
        self.adv_dataset = DADAE_random_dataset(args.data_dir,args.apps_to_leave_out, type='adv')
        self.adv_dataloader = DataLoader(self.adv_dataset, batch_size=args.batch_size_adv, shuffle=True, num_workers=4)
        # be careful to config the two batch_size args correctly, such that the num_folds are the same
        # num_folds = len(dae_dataset) // batch_size_dae  = len(adv_dataset) // batch_size_adv
    
    def _get_parameters(self, initial_lr):
        params = [
            {"params": self.encoder.parameters(), "lr": 0.2 * initial_lr},
            {"params": self.decoder.parameters(), "lr": 1.0 * initial_lr},
            {'params': self.adaptation_criterion.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        ]

    def _get_default_logger(self):
        logfile = "./log/DADAE.log"
        make_file_dir(logfile)
        logger = logging.getLogger("DADAE logger")
        logger.setLevel(logging.DEBUG)
        ch = logging.FileHandler(logfile)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def _train(self, epoch):
        self.model.train()
        idx = 0
        for (source ,label), (target,_) in zip(self.dae_dataloader,self.adv_dataloader):
            source = source.to(self.device)
            label = label.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            source_output = self.encoder(source)
            reconstructed_input = self.decoder(source_output)
            target_output = self.encoder(target)

            loss = self.reconstruction_criterion(reconstructed_input, label) + self.args.alpha * self.adaptation_criterion(source_output, target_output)
            loss.backward()
            self.optimizer.step()
            if idx % self.args.log_interval == 0:
                self.logger.info("Train Epoch {}, [{} / {}]|[{} / {}], Loss: {:.6f}".format(
                    epoch, idx * len(source), len(self.dae_dataset),idx * len(target), len(self.adv_dataset), loss.item()
                ))
    
    def _test(self):
        OUTPUT = []
        LABELS= []
        self.model.eval()
        test_loss = 0.
        with torch.no_grad():
            for data, label in self.adv_dataloader:
                data = data.to(self.device)
                output = self.encoder(data)
                output = self.decoder(output)
                label = label.to(self.device)
                loss = self.reconstruction_criterion(output, label)
                test_loss += loss.item()
                OUTPUT.append(output.float().cpu().numpy()[0])
                LABELS.append(label.cpu().numpy()[0])

        test_loss /= len(self.dataset)

        self.logger.info("Test Avg loss: {:.4f}".format(test_loss))
        return test_loss, OUTPUT, LABELS
    
    def save_model_state_dict(self,save_filename):
        make_file_dir(save_filename)
        torch.save(self.model.state_dict(), save_filename)

    def trainer(self):
        for epoch in range(self.args.epochs):
            self._train(epoch)
            test_loss, OUTPUT, LABELS = self._test(epoch)
            # nni.report_intermediate_result(test_loss)
        if self.save_model:
            self._save_model_state_dict()
        # nni.report_final_result(test_loss)
