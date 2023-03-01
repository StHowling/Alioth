import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from utils import *
import nni
import argparse
import logging 
import torch.optim as optim
from easydict import EasyDict
from tqdm import tqdm   
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB


class pretrained_encoder(nn.Module):
    def __init__(self,args):
        super(pretrained_encoder,self).__init__()
        self.encoder=nn.ModuleList()
        self.encoder.append(nn.Linear(args.input_size, args.encoder_sizes[0]))
        self.encoder.append(nn.ReLU())
        for i in range(len(args.encoder_sizes)-1):
            self.encoder.append(nn.Linear(args.encoder_sizes[i], args.encoder_sizes[i+1]))
            self.encoder.append(nn.ReLU())

        # self.decoder=nn.ModuleList()
        # self.decoder.append(nn.Linear(args.encoder_sizes[-1],args.decoder_sizes[0]))
        # self.decoder.append(nn.ReLU())

        # for i in range(len(args.decoder_sizes)-1):
        #     self.decoder.append(nn.Linear(args.decoder_sizes[i], args.decoder_sizes[i+1]))
        #     self.decoder.append(nn.ReLU())
        # self.decoder.append(nn.Linear(args.decoder_sizes[-1], args.ouput_size))
        # self.decoder.append(nn.ReLU())

        
    def forward(self,x):
        for idx, layer in enumerate(self.encoder):
            x=layer(x)
        return x

    def load_partial_params(self,modelfile):
        pretrained_dict=torch.load(modelfile)
        new_dict=self.state_dict()

        pretrained_dict={k: v for k, v in pretrained_dict.items() if k in new_dict}
        new_dict.update(pretrained_dict)
        self.load_state_dict(new_dict)


class CL_classifier(nn.Module):
    def __init__(self,args):
        super(CL_classifier, self).__init__()

        pretrained_model=pretrained_encoder(args)
        pretrained_model.load_partial_params(args.pretrained_model_path)

        self.encoder=pretrained_model
        if args.sole_classifier!=0:
            for param in self.encoder.parameters():
                param.requires_grad=False
        
        self.classifier=nn.ModuleList()

        self.classifier.append(nn.Linear(args.encoder_sizes[-1],args.classifier_sizes[0]))
        # self.classifier.append(nn.Linear(args.output_size,args.classifier_sizes[0]))
        self.classifier.append(nn.ReLU())

        for i in range(len(args.classifier_sizes)-1):
            self.classifier.append(nn.Linear(args.classifier_sizes[i], args.classifier_sizes[i+1]))
            self.classifier.append(nn.ReLU())

        self.classifier.append(nn.Linear(args.classifier_sizes[-1],args.final_output_size))
        self.classifier.append(nn.ReLU())
        
        def forward(self, x):
            x=self.encoder(x)
            for idx, layer in enumerate(self.classifier):
                x=layer(x)
            return x

class CLC_dataset(Dataset):
    def __init__(self, df):
        self.raw_data=df
        self.raw_data=torch.from_numpy(self.raw_data.values).float()

        self.x=self.raw_data[:,:-1]
        self.label=self.raw_data[:,-1].long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.label[idx]


class CL_classifier_trainer():
    def __init__(self,args,logger):
        self.args=args
        self.logger=logger
        self.use_cuda=args.cuda and torch.cuda.is_available()
        self.device=torch.device("cuda" if self.use_cuda else "cpu")
        self.model=CL_classifier(args).to(self.device)

        self.criterion=nn.CrossEntropyLoss().to(self.device)
        self.optimizer=optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), 
                                    lr=args.lr, weight_decay=args.weight_decay)
        self.save_model=args.save_model

        self.dataset=pd.read_csv(args.data_dir)
        self.train_dataset, self.test_dataset=train_test_split(self.dataset.values, testsize=args.test_ratio,random_state=99)
        self.train_dataset=pd.DataFrame(self.train_dataset,columns=self.dataset.columns)
        self.test_dataset=pd.DataFrame(self.test_dataset,columns=self.dataset.columns)

        if len(args.dropped_categories) > 0:
            dc=[category_map[i] for i in args.dropped_categroies]
            self.train_dataset = self.train_dataset[~self.train_dataset['CATEGORY'].isin(dc)]
            self.train_dataset=relabel(self.train_dataset,args.dropped_categories)

            for item in dc:
                self.test_dataset.loc[self.test_dataset['CATEGORY']==item,['CATEGORY']]==-1
            self.test_dataset=relabel(self.train_dataset,args.dropped_categories)

        self.train_dataloader=DataLoader(CLC_dataset(self.train_dataset),batch_size=args.batch_size,drop_last=True,shuffle=True)
        self.test_dataloader=DataLoader(CLC_dataset(self.test_dataset),batch_size=args.batch_size,drop_last=True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=len(self.train_dataloader),eta_min=0,last_epoch=-1)

        # Initialize the confidence predictor

        # SC' 12 version
        feature_nums=self.train_dataset.values.shape[1]-1
        self.distribution_matrix=np.zeros((feature_nums,2))
        gmm=GaussianMixture(n_components=1)
        for i in range(feature_nums):
            gmm.fit(self.train_dataset.values[:,i].reshape(-1,1))
            self.distribution_matrix[i,0]=gmm.means_[0,0]
            self.distribution_matrix[i,1]=gmm.covariances_[0,0,0]

        # self.distribution_matrix_1=np.zeros((feature_nums,2))
        # for i in range(feature_nums):
            # self.distribution_matrix[i,0]=self.train_dataset.values[:,i].mean()
            # self.distribution_matrix[i,1]=self.train_dataset.values[:,i].var()
        # print(self.distribution_matrix-self.distribution_matrix_1)
        '''Basically no difference'''

        # Leveraging CL representation
        # known_hidden_vectors=self.train_dataset.values[:,:-1]
        # self.model.eval()
        # known_hidden_vectors=self.model.encoder(torch.from_numpy(known_hidden_vectors).float().to(self.device)).to(torch.device('cpu')).detach().numpy()
        
        # self.confidence_predictor=GaussianNB()
        # self.confidence_predictor=GaussianMixture(n_components=len(self.train_dataset['CATEGORY'].unique()))
        # self.confidence_predictor.fit(known_hidden_vectors,self.train_dataset.values[:,-1].astype(float))

    def train(self):
        self.logger.info(f"Start CL classifier training for {self.args.epochs} epochs.")
        n_iter=0

        self.model.train()
        best_acc, best_epoch=0,0
        for epoch_counter in range(self.args.epochs):
            train_loss, correct=0., 0.
            idx=0
            for vm_stats, labels in tqdm(self.train_dataloader):
                VM, labels= vm_stats.to(self.device), labels.to(self.device)

                output=self.model(VM)
                loss=self.criterion(output,labels)

                train_loss+=loss.item()
                correct+=(output.argmax(1)==labels).type(torch.float).sum().item()
                # correct+=(F.softmax(output).argmax(1)==labels).type(torch.float).sum().item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if n_iter % self.args.log_interval == 0:
                    test_acc, known_acc, unknown_acc = self._test()
                    self.logger.info("Train Epoch {}. [{} / {}], Loss: {:.6f}, Acc/Test_total: {:.6f}, Acc/Test_unknown: {:.6f}".format(epoch_counter, idx*len(VM)//2, len(self.train_dataset), loss.item(), test_acc, known_acc, unknown_acc))

                n_iter+=1
                idx+=1
            
            if epoch_counter>=10:
                self.scheduler.step()
                if test_acc>best_acc:
                    test_acc=best_acc
                    best_epoch=epoch_counter
                    if self.save_model:
                        self._save_model_state_dict()

            train_loss/=len(self.train_dataloader)
            train_acc= correct/ len(self.train_dataloader.dataset)
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {train_loss}\tTrain accuracy: {train_acc}\tTest accuracy: {test_acc}")
            nni.report_intermediate_result(test_acc)

        self.logger.info(f"Best accuracy: {best_acc}\tAchieved Epoch: {best_epoch}")
        nni.report_final_result(test_acc)


    def check_distribution(self,vm_stat,threshold=197):
        '''Highly screwed cherry-picking use of confidence predictor'''
        mean=torch.from_numpy(self.distribution_matrix[:,0])
        mean=torch.cat([mean for i in range(vm_stat.shape[0])],dim=0).view(vm_stat.shape[0],-1)

        var=torch.from_numpy(self.distribution_matrix[:,1])*2
        var=torch.cat([var for i in range(vm_stat.shape[0])],dim=0).view(vm_stat.shape[0],-1)

        diff=abs(vm_stat-mean)
        confidence_result=((diff>=var).type(torch.float).sum(dim=1)>threshold).long()

        return confidence_result
        

    def _test(self):
        self.model.eval()

        known_correct, unknown_correct, unknown_samples = 0., 0., 0.
        # test_loss=0.

        with torch.no_grad():
            # known_hidden_vectors=self.model.encoder(torch.from_numpy(known_hidden_vectors).float().to(self.device)).to(torch.device('cpu')).detach().numpy()
            # self.confidence_predictor.fit(known_hidden_vectors,self.train_dataset.values[:,-1].astype(float))
            for VM, labels in self.test_dataloader:
                tmp=self.check_distribution(VM)
                VM,labels=VM.to(self.device),labels.to(self.device)
                output=self.model(VM)
                output=F.softmax(output,dim=1)

                '''choose one, but neither is good'''
                # hidden_vector=self.model.encoder(VM)
                # hidden_vector = VM

                # confidence_output=self.confidence_predictor.predict_proba(hidden_vector.to(torch.device('cpu')).numpy())
                # confidence_output=torch.from_numpy(confidence_output)

                # loss=self.criterion(output, labels)
                # test_loss+=loss.item()

                prediction=output.argmax(1)
                unknown_samples+=(labels==-1).type(torch.float).sum.item()


                # known_correct += (output.argmax(1) == labels).type(torch.float).sum().item()
                # unknown_samples += (labels == -1).type(torch.float).sum().item()

                # tmp = (output.max(1)[0]<=0.5).type(torch.int)*(-1)
                # tmp = (confidence_output.max(1)[0]<=0.5).type(torch.int)*(-1)

                for i in range(output.shape[0]):
                    if tmp[i]==1 and labels[i]==-1:
                        unknown_correct+=1.0
                    elif tmp[i]!=1 and prediction[i]==labels[i]:
                        known_correct+=1.0

        # test_loss /= len(self.test_dataloader)
        total_samples=len(self.test_dataloader)*self.args.batch_size
        acc = (known_correct+unknown_correct)/total_samples
        known_acc = known_correct / (total_samples-unknown_samples)
        unknown_acc = 0.
        if unknown_samples>0:
            unknown_acc = unknown_correct / unknown_samples

        self.logger.info("Correct Known: {} / {}\tCorrect Unknown: {} / {}".format(known_correct,total_samples-unknown_samples,unknown_correct,unknown_samples))

        return acc,known_acc,unknown_acc

    def _save_model_state_dict(self):
        save_filename=self.args.save_filename
        make_file_dir(save_filename)
        model_st=self.model.state_dict()
        
        torch.save(model_st, save_filename)

def get_parameters():
    parser=argparse.ArgumentParser(description="Contrasitive Learner")
    parser.add_argument("--data_dir",type=str,
                        default='../data/raw-data/cl_data.csv',help="data file path")
    
    parser.add_argument("--config",type=str,
                        default="../config/CL_classifier_config.json",help="config file path")
    parser.add_argument("--pretrained_model_path", type=str, default='../model/cl_test.pt', 
                        help="Pretrained CL backbone model file")

    parser.add_argument("--save_model",type=int,default=0,metavar='N',help="Whether to save model")
    parser.add_argument("--batch_size",type=int,default=128,metavar='N',
                        help="input batch size for training (default: 100)")

    parser.add_argument('--lr',type=float,default=0.001,metavar='LR',help="learning rate (default: 0.0003)")
    parser.add_argument('--wd','--weight-decay',default=1e-4,type=float,
                        metavar='W',help="weight decay (default: 1e-4)",dest='weight_decay')
    
    parser.add_argument('--epochs',type=int,default=100,metavar='N',
                        help="number of epochs to train (default:100)")
    parser.add_argument('--cuda', type=int, default=1, metavar='N',help="use CUDA training")

    parser.add_argument('--drop_n',default=0,type=int,metavar='N',
                        help="Number of categories to drop randomly and act as unknown classes")
    parser.add_argument('--log_interval',type=int,default=100,metavar='N',
                        help='log interval in the number of batches')
    
    parser.add_argument('--save_filename',type=str,default="../model/CLC_model0.pt",
                        help="model name to save")

    parser.add_argument('--sole_classifier', type='int', default=0, metavar='N', 
                        help="whether to freeze the pretrained encoder and only train classifier")
    args=parser.parse_args()
    return args


if __name__=='__main__':
    logfile='./log/CL_classifier.log'
    make_file_dir(logfile)
    logger=logging.getLogger("CL classifier Logger")
    logger.setLevel(logging.DEBUG)
    ch=logging.FileHandler(logfile)
    ch.setLevel(logging.DEBUG)
    formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    cmd_args=EasyDict(vars(get_parameters()))
    args=load_args("../config/CL_classifier_config.json")
    tuner_args=nni.get_next_parameter()
    args=merge_parameter(args,cmd_args)
    args=merge_parameter(args,tuner_args)
    logger.info(args)

    CL_classifier_trainer(args).train()
