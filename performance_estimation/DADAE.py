'''
Adpated from Jindong Wang's famous transferlearning repo on Github:
https://github.com/jindongwang/transferlearning/blob/master/code/DeepDA
'''
from typing import Any, Mapping
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

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