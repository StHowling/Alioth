import numpy as np
import pandas as pd


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
# from torch.cuda.amp import GradScaler, autocast

from utils import *
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import nni
import argparse
import logging
from easydict import EasyDict
from tqdm import tqdm
import datetime

'''
This Python file includes all classes and functions used in contrastive learning phase.
To be specific, they are daaset builder, auto-encoder model, CL trainer, logger, parser and nni utilities.
This phase is unsupervised and uses the whole set as training set, i.e. no testing set.
(Maybe we can, somehow, construct a testing set that is the normal case in supervised learning, or even
specially designed for the open-set learning problem, containing unseen categories in training.)
To introduce testing into CL phase, you can try:
1. Adopt the same setting as training set: augmentation process, batch_size and info_nce_loss must be the same.
Drop_last should also be True. Be aware to test the model in batch manner since info_nce_loss is hard_coded with args.batch_size.
Without info_nce_loss we cannot compute accuracy. (Maybe you can try to decouple them)
2. Merge the two phases, alternatively update CL enc-dec and (MLP) classifier. You can choose to fix or not CL encoder
when updating the classifier via nn.CrossEntropyLoss. Be aware that CL phase uses augmented data while classification phase
uses original features.

To introduce 'unknown' categories into the two phases (you cannot introduce them into only 1 phase), specify
the drop_n (random) or dropped_categories (selected) parameters in configuration.
For 'unknown' categories in the CL phase, currently we just drop them in training set. If you want to introduce
testing into CL phase, retain them in the testing set.

Current to-do list mainly focus on the open-set learning (identifying unknown categories), since the method is highly accurate,
but cannot learn a represenation that suits for distinguishing unseen classes. What the classifier learns is not as expected,
the result is only a score for each seen class rather than probability. The scores are not normalized, as high as several hundred
in scale, and softmax makes no sense--softmax assumes that sum(probaility of seen classes)=1, but it should be <<1.

1. Introduce testing and unknown categories into CL phase. See if the model can correctly distinguish the unknown categories with others
without seeing them in training. Use visualization (T-SNE with PCA initialization).
2. Merge the two phases and see if the learned representation improves in terms of distinguishing unknown categories.
(Use GMM mixture or visualization)
3. Perform feature selection and see if it agrees with the estimation models' results. Check if the 3 methods 
(clustering on learned representation, clustering on original features, confidence predictor in Practical Method)
improves with less features.
4. Use other open-set learning paradigms.
5. Try to switch to CNN/RNN/Tranformer structures. Change the data schema if needed. 
    - RNN/Transformer: N*T*h (N=batch_size, T=window_length, h=feature_num)
    - CNN: N*h*h (get an h*h tensor through matrix multiplication / cross product)
'''

# Augmentation Class

class DataAugmenter:
    '''
    Given an N x h np.ndarray object as dataset (raw), randomly generates N_views (default=2) augmented samples from 
    each sample (1 row). N is the total number of samples, h is the number of input features (excluding labels).
    The augmenter assumes that all samples it can see belong to the same class, and we will build 1 augmenter for each
    known class when constructing CL_random_dataset.
    Current implemented augmentation method is either adding random noise or using SMOTE, not sequentially applying both 
    as in original simCLR. This should be checked--which is better. Current level of random noise is also arbitrary,
    which needs experiments to determine the appropriate level, and if we need some adaptive mechanism.
    '''
    def __init__(self,dataset,mu=0,sigma=None, N=2,k=5):
        self.mu=mu
        self.N=N
        self.K=k
        if type(dataset)!= type(np.zeros(1)):
            raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                ('dataset', type(np.zeros(1)), type(dataset))
            )
        self.data=dataset.astype('float32')

        if sigma==None:
            self.sigma=np.var(self.data,axis=0)
        else:
            self.sigma=sigma
        self.knn=NearestNeighbors(n_neighbors=self.K).fit(self.data)

    def get_augmented(self,x):
        synthesized=np.zeros((self.N,self.data.shape[1]))
        for i in range(self.N):
            choice=np.random.rand()
            if choice>=0.5:
                # return a 1d array containing indices in the original dataset of K nearest neighbors of X
                nn_array=self.knn.kneighbors(x.reshape(1,-1),return_distance=False)[0]

                # select 1 neighbor randomly
                nn=np.random.randint(0,self.K-1)

                # randomly select a point between x and its neighbor
                diff=self.data[nn_array[nn]]-x 
                gap=np.random.rand()
                synthesized[i]=x+diff*gap
            else:
                # apply a random noise on x, whose variance is determined based on the category's distribution
                synthesized[i]=x+np.random.normal(self.mu,self.sigma)

        return synthesized


class ContrastiveLearner(nn.Module):
    '''
    Currently just an autoencoder module based on MLP. Decoder is a 2/3 layer projection head.
    output_size should not be 1. 
    ReLU is better than Sigmoid. Adam is better than SGD.
    lr_scheduler makes it more stable and decrease final loss.
    For 218 features,  parameter search shows that output_size=64/32 is appropriate.
    Recommended setting: [256,128][64]-64 / [256,128,128][128,32]-32
    Shallower net performs better for final classification.
    '''
    def __init__(self,args):
        super(ContrastiveLearner,self).__init__()
        self.encoder=nn.ModuleList()
        self.encoder.append(nn.Linear(args.input_size, args.encoder_sizes[0]))
        self.encoder.append(nn.ReLU())
        
        for i in range(len(args.encoder_sizes)-1):
            self.encoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i+1]))
            self.encoder.append(nn.ReLU())
        
        self.decoder=nn.ModuleList()
        self.decoder.append(nn.Linear(args.encoder_sizes[-1],args.decoder_sizes[0]))
        self.decoder.append(nn.ReLU())

        for i in range(len(args.decoder_sizes)-1):
            self.decoder.append(nn.Linear(args.decoder_sizes[i], args.decoder_sizes[i+1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(args.decoder_sizes[-1], args.ouput_size))
        self.decoder.append(nn.ReLU())

    def forward(self, x):
        for idx, layer in enumerate(self.encoder):
            x=layer(x)
        for idx, layer in enumerate(self.decoder):
            x=layer(x)
        return x



class CL_random_dataset(Dataset):
    '''
    Generate the augmented dataset from the original dataset. First for each category, initialize an augmenter.
    Then for each data point, call the corresponding augmenter to get augmented data. Finally encapsulate all of them.
    Requires that the passed-in dataframe df only contains input features and the 'CATEGORY' column.
    '''
    def __init__(self,df,N_views=2):
        self.raw_data=df
        self.categories=pd.unique(self.raw_data['CATEGORY'])

        self.data_augmenters={}
        for item in self.categories:
            tmp=self.raw_data[self.raw_data['CATEGORY']==item].copy()
            tmp.drop(['CATEGORY'],axis=1,inplace=True)
            self.data_augmenters[item]=DataAugmenter(tmp.values,N=N_views)

        self.x=[]
        for i in range(self.raw_data.shape[0]):
            augmented_x=self.data_augmenters[self.raw_data.iloc[i,-1]].get_augmented(self.raw_data.iloc[i,:-1].values.astype('float32'))
            self.x.append([augmented_x[j] for j in range(N_views)])
        self.x=np.array(self.x)
        self.x=torch.from_numpy(self.x).float()

        del self.raw_data
        for item in self.categories:
            del self.data_augmenters[item]
        del self.data_augmenters

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index]

def accuracy(output, target, topk=(1,)):
    # Computes the accuracy over the k top predictions for the specified values of k (fetched from SimCLR)
    # Comments by Sthowling
    with torch.no_grad():
        maxk=max(topk)
        batch_size=target.size(0)

        _,pred=output.topk(maxk,1,True,True)
        # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor) (val, index)
        # output: batch_size x batch_size tensor, predicted similarity with other samples
        # pred: batch_size x k tensor, each row corresponds to the top-k predicted label of a sample

        pred=pred.t()
        # k x batch_size, the i-th row is the predicted i-th most possible positive pair candidate

        correct=pred.eq(target.view(1,-1).expand_as(pred))
        # view: resize the label tensor as 1 x batch_size, previously 1-d, (batch_size, )
        # expand_as: duplicate the tensor k times to match the size of pred (k x batch_size)
        # each row is the same and is the label of this batch
        # eq: element-wise comparison, output a k x batch_size Boolean matrix; each column has at most 1 'True'
        # correct[i][j]=True means that for j-th sample in this batch, its prediction is correct for top-k (k>=i)
        # metrics but incorrect for top-k (k<i) metrics, i.e. the real label is predicted as the i-th most possible
        # positive pair candidate for sample j

        res=[]
        # return an array with the same shape of the tuple topk, where each element i represents the accuracy when the metric is top-topk[i]
        for k in topk:
            correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
            # reshape: get a (k x batch_size, ) 1-d tensor
            # float: convert Boolean to real
            # keepdim=True holds the 1-d tensor data structure, get a (1, ) for the number of correctly classified samples

            res.append(correct_k.mul_(100.0/batch_size))
        return res


# Train a contrastive learner

class CL_trainer():
    def __init__(self,args,logger):
        self.args=args
        self.logger=logger
        if args.drop_n!=0 and len(args.dropped_categories)!=0:
            raise ValueError('Both random dropping and selected dropping is configured. Check \'args.drop_n\' and \'args.dropped_categories\'.')
        self.use_cuda=args.cuda and torch.cuda.is_available()
        self.device=torch.device("cuda" if self.use_cuda else "cpu")
        self.model=ContrastiveLearner(args).to(self.device)

        self.criterion=nn.CrossEntropyLoss().to(self.device)
        self.optimizer=optim.Adam(self.model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        self.save_model=args.save_model

        df=pd.read_csv(args.data_dir)
        if args.drop_n!=0:
            df,self.dropped_categories=df_drop_random_categories(df,args.drop_n)
        elif len(args.dropped_categories)!=0:
            df=df_drop_selected_categories(df,args.dropped_categories)

        self.dataset=CL_random_dataset(df,args.n_views)
        self.train_dataloader=DataLoader(self.dataset,batch_size=args.batch_size, drop_last=True, shuffle=True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=len(self.train_dataloader),eta_min=0,last_epoch=-1)

    def info_nce_loss(self,features):
        '''
        Implement the NCE loss in the original paper (fetched from SimCLR)
        '''
        labels=torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)],dim=0)
        # assert self.args.n_views=2

        labels=(labels.unsqueeze(0)==labels.unsqueeze(1)).float()
        labels=labels.to(self.device)

        features=F.normalize(features,dim=1)

        similarity_matrix=torch.matmul(features,features.T)
        # assert similarity_matrix.shape == (
        #   self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # == labels.shape

        # discard the main diagonal from both: labels and similariy matrix
        mask=torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels=labels[~mask].view(labels.shape[0],-1)
        similarity_matrix=similarity_matrix[~mask].view(similarity_matrix.shpae[0],-1)

        positives=similarity_matrix[labels.bool()].view(similarity_matrix.shape[0],-1)
        negatives=similarity_matrix[~labels.bool()].view(similarity_matrix.shpae[0],-1)

        logits=torch.cat([positives,negatives],dims=1)
        labels=torch.zeros(logits.shape[0],dtype=torch.long).to(self.device)

        logits=logits/self.args.temperature
        return logits, labels

    def train(self):
        self.logger.info(f"Start CL training for {self.args.epochs} epochs.")
        # drop categories according to configuration
        if self.args.drop_n !=0:
            dropped_info=','.join(self.dropped_categories)
            self.logger.info(f"Dropping {dropped_info} in this experiment.")
        elif len(self.args.dropped_categories)!=0:
            dropped_info=','.join(self.args.dropped_categories)
            self.logger.info(f"Dropping {dropped_info} in this experiment.")

        n_iter=0
        best_loss, best_epoch=1e9,0

        for epoch_counter in range(self.args.epochs):
            idx=0

            for vm_stats in tqdm(self.train_dataloader):
                VM=torch.cat([vm_stats[:,i,:] for i in range (vm_stats.shzpe[1])],dim=0)
                VM=VM.to(self.device)

                output=self.model(VM)

                logits, labels = self.info_nce_loss(output)
                loss = self.criterion(logits,labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.log_every_n_steps == 0:
                    top1 = accuracy(logits, labels)
                    self.logger.info("Train Epoch {}, [{} / {}], Loss: {:.6f}, Acc/Top1: {:.6f}".format(
                        epoch_counter,  idx * len(VM)//2, len(self.train_dataset), loss.item(), top1[0] 
                    ))
                
                n_iter+=1
                idx+=1
            
            epoch_loss=loss.item()
            if epoch_counter>=10:
                self.scheduler.step()
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                    best_epoch=epoch_counter
                    if self.save_model:
                        self._save_model_state_dict()

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss.item()}\tTop1 accuracy: {top1[0]}")
            nni.report_intermediate_result(loss.item())
        
        self.logger.info(f"Best loss: {best_loss}\tAchieved Epoch: {best_epoch}")
        nni.report_final_result(loss.item())
        

    def _save_model_state_dict(self):
        make_file_dir(self.args.save_filename)

        model_st=self.model.state_dict()
        torch.save(model_st, self.args.save_filename)
        
def get_parameters():
    parser=argparse.ArgumentParser(description="Contrasitive Learner")
    parser.add_argument("--data_dir",type=str,
                        default='../data/raw-data/cl_data.csv',help="data file path")
    
    parser.add_argument("--config",type=str,
                        default="../config/CL_config.json",help="config file path")
    parser.add_argument("--save_model",type=int,default=0,metavar='N',help="Whether to save model")
    parser.add_argument("--batch_size",type=int,default=32,metavar='N',
                        help="input batch size for CL training (default: 32)")

    parser.add_argument('--lr',type=float,default=0.001,metavar='LR',help="learning rate (default: 0.0003)")
    parser.add_argument('--wd','--weight-decay',default=1e-4,type=float,
                        metavar='W',help="weight decay (default: 1e-4)",dest='weight_decay')
    
    parser.add_argument('--epochs',type=int,default=200,metavar='N',
                        help="number of epochs to train (default:200)")
    parser.add_argument('--cuda', type=int, default=1, metavar='N',help="use CUDA training")

    parser.add_argument('--drop_n',default=0,type=int,metavar='N',
                        help="Number of categories to drop randomly and act as unknown classes")
    parser.add_argument('--log_interval',type=int,default=100,metavar='N',
                        help='log interval in the number of batches')
    
    parser.add_argument('--save_filename',type=str,default="../model/CL_model0.pt",
                        help="model name to save")
    args=parser.parse_args()
    return args

if __name__=='__main__':
    logfile='./log/CL.log'
    make_file_dir(logfile)
    logger=logging.getLogger("CL Logger")
    logger.setLevel(logging.DEBUG)
    ch=logging.FileHandler(logfile)
    ch.setLevel(logging.DEBUG)
    formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    cmd_args=EasyDict(vars(get_parameters()))
    args=load_args("../config/CL_config.json")
    tuner_args=nni.get_next_parameter()
    args=merge_parameter(args,cmd_args)
    args=merge_parameter(args,tuner_args)
    logger.info(args)

    CL_trainer(args,logger).train()