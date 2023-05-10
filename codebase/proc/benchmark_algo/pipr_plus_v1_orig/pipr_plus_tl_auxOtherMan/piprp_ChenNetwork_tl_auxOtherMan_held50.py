#Based on paper Multifaceted protein–protein interaction prediction based on Siamese residual RCNN by Chen, Ju, Zhou, Chen, Zhang, Chang, Zaniolo, and Wang
#https://github.com/muhaochen/seq_ppi

import os, sys
# #add parent and grandparent to path
# currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(currentdir)
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)
# parentdir = os.path.dirname(parentdir)
# sys.path.append(parentdir)

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import torch
from utils_benchmark.NetworkRunnerCollate import NetworkRunnerCollate
from proc.benchmark_algo.pipr_plus.GenericNetworkModel_piprp import GenericNetworkModel
from proc.benchmark_algo.pipr_plus.GenericNetworkModule_piprp import GenericNetworkModule
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn import preprocessing


class ChenNetwork(nn.Module):
    def __init__(self,hiddenSize=50,inSize=14,aux_oneDencodingsize=1024,numLayers=6,seed=1):
        super(ChenNetwork, self).__init__()
        self.pooling = nn.MaxPool1d(3)
        self.activation = nn.LeakyReLU(0.3)
        self.numLayers = numLayers
        torch.manual_seed(seed)

        self.convLst = nn.ModuleList()
        self.GRULst = nn.ModuleList()
        for i in range(0,self.numLayers):
            if i == 0: #first convolutions takes data of input size, other 5 take data of hidden size * 3
                self.convLst.append(nn.Conv1d(inSize,hiddenSize,3))
            else:
                self.convLst.append(nn.Conv1d(hiddenSize*3,hiddenSize,3))
            if i<= 4: #only numlayers-1 grus
                self.GRULst.append(nn.GRU(input_size=hiddenSize,hidden_size=hiddenSize,bidirectional=True,batch_first=True))
        
        self.linear1 = nn.Linear(2*(hiddenSize+aux_oneDencodingsize),100)
        self.linear2 = nn.Linear(100,(hiddenSize+7)//2)
        self.linear3 = nn.Linear((hiddenSize+7)//2,2)
        
    def forward(self,x):
        (protA, protB, auxProtA, auxProtB) = x
        protLst = []
        for item in [protA, protB]: #run each protein through gru/pooling layers
            for i in range(0,self.numLayers-1):
                #conv1d and pooling expect hidden dim on 2nd axis (dim=1), gru needs hidden dim on 3rd axis (dim=2) . . .
                item = item.permute(0,2,1)
                item = self.convLst[i](item)
                item = self.pooling(item)
                item = item.permute(0,2,1)
                item2,hidden = self.GRULst[i](item)
                item = torch.cat((item,item2),2)
            
            item = item.permute(0,2,1)
            item = self.convLst[self.numLayers-1](item)
            item = item.mean(dim=2) #global average pooling over dim 2, reducing the data from 3D to 2D
            protLst.append(item)
        
        protA = protLst[0]
        protB = protLst[1]

        # now horizontally concatenate, auxProtA with protA and auxProtB with protB
        concat_protA = torch.cat((protA, auxProtA), 1)
        concat_protB = torch.cat((protB, auxProtB), 1)

        # x = torch.mul(concat_protA,concat_protB)  # element wise multiplication
        x = torch.cat((concat_protA, concat_protB), dim=1)  # side-by-side concatenation 
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
    
class NetworkRunnerChen(NetworkRunnerCollate):
    def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=1,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={},skipScheduler=30):
        NetworkRunnerCollate.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
        self.skipScheduler = hyp.get('skipScheduler',skipScheduler)
    
            
    def updateScheduler(self,values):
        if self.scheduler is not None and self.epoch > self.skipScheduler:
            self.scheduler.step(values)

class ChenModel(GenericNetworkModel):
    def __init__(self,hyp={},inSize=12,aux_oneDencodingsize=1024,hiddenSize=50,numLayers=6,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=5e-4,minLr=1e-4,schedFactor=.5,schedPatience=3,schedThresh=1e-2,threshSchedMode='abs'):
        GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)

        self.inSize = inSize
        self.aux_oneDencodingsize = aux_oneDencodingsize
        self.hiddenSize = hyp.get('hiddenSize',hiddenSize)
        self.numLayers = hyp.get('numLayers',numLayers)

        #move uncommon network runner properties into hyperparams list if needed
        hyp['amsgrad'] = hyp.get('amsgrad',True)
        hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
        
        
    def genModel(self):
        self.net = ChenNetwork(self.hiddenSize,self.inSize,self.aux_oneDencodingsize,self.numLayers,self.seed)
        #self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
        self.model = NetworkRunnerChen(self.net,hyp=self.hyp,skipScheduler=self.skipScheduler)

    #train network
    def fit(self,pairLst,classes,dataMatrix,oneDdataMatrix,validationPairs=None, validationClasses=None):
        self.skipScheduler = 250000//classes.shape[0]
        super().fit(pairLst,classes,dataMatrix,oneDdataMatrix,validationPairs,validationClasses)


#protein length should be at least 3**5 to survive 5 sets of maxpool(3) layers
class ChenNetworkModule(GenericNetworkModule):
    def __init__(self, hyperParams = {}, maxProteinLength=2000, hiddenSize=50,inSize=12, aux_oneDencodingsize=1024):
        GenericNetworkModule.__init__(self,hyperParams)
        self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
        self.inSize = self.hyperParams.get('inSize',inSize) #temporary value, until data is loaded in loadFeatureData function
        self.aux_oneDencodingsize = self.hyperParams.get('aux_oneDencodingsize',aux_oneDencodingsize) #temporary value, until data is loaded in loadFeatureData function
        self.hiddenSize = self.hyperParams.get('hiddenSize',hiddenSize)
        
    def genModel(self):
        self.model = ChenModel(self.hyperParams,self.inSize,self.aux_oneDencodingsize,self.hiddenSize)
        
    def loadFeatureData(self,featureFolder):
        print('\n#### loading tl-based 2d embeddings ####')
        print("loading human_seq_feat_2d_reg_dict ...")
        # load the tl-embeddings stored in human_seq_feat_2d_reg_dict
        human_seq_feat_2d_reg_dict_pkl_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data', 'human_seq_feat_2d_reg_dict.pkl')
        human_seq_feat_2d_reg_dict = joblib.load(human_seq_feat_2d_reg_dict_pkl_path)
        print("loaded human_seq_feat_2d_reg_dict ...")

        allProteinsList = list(human_seq_feat_2d_reg_dict.keys())
        self.encodingSize = self.inSize = human_seq_feat_2d_reg_dict[int(allProteinsList[0])].shape[1]
        self.aux_oneDencodingsize = 1854  # at first run, it will fail but after observing 'aux_oneDencodingsize' print in the for loop below, its
                                          # proper value can be updated here, so that the rerun would be error free.

        self.dataLookup = {}
        self.dataMatrix = torch.zeros((len(allProteinsList),self.maxProteinLength,self.encodingSize))
        self.oneDdataMatrix = torch.zeros((len(allProteinsList), self.aux_oneDencodingsize))

        print('\n#### loading other manual 1d embeddings ####')
        print("loading human_seq_manual_feat_dict ...")
        human_seq_manual_feat_dict = joblib.load(os.path.join(Path(__file__).parents[5], 'dataset/preproc_data','human_seq_manual_feat_dict.pkl'))
        # trimming human_seq_manual_feat_dict so that it occupies less memory (RAM)
        for prot_id in list(human_seq_manual_feat_dict.keys()):
            human_seq_manual_feat_dict[prot_id]['seq'] = None
        print("loaded human_seq_manual_feat_dict ...")

        for item in allProteinsList:
            item = str(item)
            self.dataLookup[item] = len(self.dataLookup)
            self.dataMatrix[self.dataLookup[item]] = torch.from_numpy(human_seq_feat_2d_reg_dict[int(item)])

            # extract other manual 1d embeddings 
            seq_manual_feat_dict = human_seq_manual_feat_dict[int(item)]['seq_manual_feat_dict']
            other_man_feat_lst = seq_manual_feat_dict['AC30'] + seq_manual_feat_dict['PSAAC15'] + seq_manual_feat_dict['ConjointTriad'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] + seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] \
                                + seq_manual_feat_dict['CHAOS'] \
                                + seq_manual_feat_dict['AAC20'] + seq_manual_feat_dict['AAC400'] \
                                + seq_manual_feat_dict['Grantham_Sequence_Order_30'] + seq_manual_feat_dict['Schneider_Sequence_Order_30'] \
                                + seq_manual_feat_dict['Grantham_Quasi_30'] + seq_manual_feat_dict['Schneider_Quasi_30'] + seq_manual_feat_dict['APSAAC30_2']
                                # + seq_manual_feat_dict['DuMultiCTD_C'] + seq_manual_feat_dict['DuMultiCTD_T'] + seq_manual_feat_dict['DuMultiCTD_D']
            other_man_feat_arr = np.array(other_man_feat_lst)
            aux_1d_tensor = torch.from_numpy(other_man_feat_arr)
            # print('aux_oneDencodingsize: ' + str(aux_1d_tensor.shape[0]))  # important print statement as its output will be used above
            self.oneDdataMatrix[self.dataLookup[item]] = aux_1d_tensor
        # end of for loop

        # perform the normalization of the auxiliary data matrix
        print('perform the normalization of the auxiliary data matrix')
        aux_data_arr = self.oneDdataMatrix.numpy()
        scaler = preprocessing.StandardScaler()
        aux_data_arr_scaled = scaler.fit_transform(aux_data_arr)
        self.oneDdataMatrix = torch.from_numpy(aux_data_arr_scaled)

