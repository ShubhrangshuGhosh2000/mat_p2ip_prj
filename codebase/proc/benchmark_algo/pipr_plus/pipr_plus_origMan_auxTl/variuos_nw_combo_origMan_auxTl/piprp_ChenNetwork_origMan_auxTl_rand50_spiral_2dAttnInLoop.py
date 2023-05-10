#Based on paper Multifaceted protein–protein interaction prediction based on Siamese residual RCNN by Chen, Ju, Zhou, Chen, Zhang, Chang, Zaniolo, and Wang
#https://github.com/muhaochen/seq_ppi

import os, sys
# # add parent and grandparent to path
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

from utils_benchmark.NetworkRunnerCollate import NetworkRunnerCollate
from proc.benchmark_algo.pipr_plus.GenericNetworkModel_piprp import GenericNetworkModel
from proc.benchmark_algo.pipr_plus.GenericNetworkModule_piprp import GenericNetworkModule
from proc.benchmark_algo.pipr_plus.positional_encoder_piprp import PositionalEncoder
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn import preprocessing
import time


class ChenNetwork(nn.Module):
    def __init__(self,hiddenSize=50,inSize=14,aux_oneDencodingsize=1024,numLayers=6,seed=1,fullGPU=False,deviceType='cpu'):
        super(ChenNetwork, self).__init__()
        torch.manual_seed(seed)
        self.pooling = nn.MaxPool1d(3)
        self.activation = nn.LeakyReLU(0.3)
        self.numLayers = numLayers
        self.fullGPU = fullGPU
        self.deviceType = deviceType

        # ############################ Multihead Attention part -start ############################
        n_heads = 1  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number. Default: 8
        max_seq_len = 700
        dropout_pos_enc = 0.10
        dropout_mha = 0.0
        # ############################ Multihead Attention part -end ############################

        self.convLst = nn.ModuleList()
        self.SEQLst = nn.ModuleList()
        self.PosEncLst = nn.ModuleList()
        for i in range(0,self.numLayers):
            if(i == 0):  # 0th layer: only conv layer and no attention layer
                self.convLst.append(nn.Conv1d(inSize,hiddenSize,3))  # first convolutions takes data of input size, other 5 take data of hidden size * 3
                self.PosEncLst.append(None)  # no Position Encoding for the 0th layer
                self.SEQLst.append(None)  # no attention for the 0th layer
            elif(i == 1): # 1th layer : both conv and attention layers
                self.convLst.append(nn.Conv1d(hiddenSize,hiddenSize*2,3))
                self.PosEncLst.append(PositionalEncoder(d_model= hiddenSize, dropout=dropout_pos_enc, max_seq_len=max_seq_len, batch_first=True))
                self.SEQLst.append(nn.MultiheadAttention(embed_dim=hiddenSize, num_heads=n_heads, dropout=dropout_mha, batch_first=True))
            elif((i > 1) and (i <= self.numLayers-2)):  # intermediate layers between first and last layer : both conv and attention layers
                self.convLst.append(nn.Conv1d(hiddenSize*(i+1),hiddenSize,3))
                self.PosEncLst.append(PositionalEncoder(d_model= hiddenSize*(i+1), dropout=dropout_pos_enc, max_seq_len=max_seq_len, batch_first=True))
                self.SEQLst.append(nn.MultiheadAttention(embed_dim=hiddenSize*(i+1), num_heads=n_heads+1, dropout=dropout_mha, batch_first=True))
            elif(i == self.numLayers -1):  # last layer: only conv layer and no attention layer
                self.convLst.append(nn.Conv1d(hiddenSize*(i+1),hiddenSize,3))
                self.PosEncLst.append(None)  # no Position Encoding for the last layer
                self.SEQLst.append(None)  # no attention for the last layer
            
        # self.linear1 = nn.Linear(2*(hiddenSize+aux_oneDencodingsize),100)
        # self.linear2 = nn.Linear(100,(hiddenSize+7)//2)
        # self.linear3 = nn.Linear((hiddenSize+7)//2,2)
        layer_1_size = 1024
        layer_2_size = 512
        layer_3_size = 256
        self.linear1 = nn.Linear(2*(hiddenSize+aux_oneDencodingsize), layer_1_size)
        self.linear2 = nn.Linear(layer_1_size, layer_2_size)
        self.linear3 = nn.Linear(layer_2_size, layer_3_size)
        self.linear4 = nn.Linear(layer_3_size, 2)
        self.bn1 = nn.BatchNorm1d(layer_1_size)
        self.bn2 = nn.BatchNorm1d(layer_2_size)
        self.bn3 = nn.BatchNorm1d(layer_3_size)
        
    def forward(self,x):
        (protA, protB, auxProtA, auxProtB) = x
        protLst = []
        for item in [protA, protB]: #run each protein through attention/pooling layers
            for i in range(0,self.numLayers):
                #conv1d and pooling expect hidden dim on 2nd axis (dim=1), attention needs hidden dim on 3rd axis (dim=2)
                # if item is (a,b,c), then input to conv is (a,c,b) and input to attention is (a,b,c) where
                # a = batch size, b = no. of time-steps or sequence length and c = no. of features per time-step or no. of channels
                if(i == 0):   # 0th layer: only conv layer and no attention layer
                    # print("From model.forward(): Size of item at beginning when i is " + str(i) + " : " + str(item.size()))
                    item = item.permute(0,2,1)  # item: (a,c,b)
                    item = self.convLst[i](item)
                    item = self.pooling(item)
                    item = item.permute(0,2,1)  # item: (a,b,c)
                    # print("From model.forward(): Size of item  at end when i is " + str(i) + " : " + str(item.size()))
                elif(i == 1):  # 1th layer : both conv and attention layers
                    # print("From model.forward(): Size of item at beginning when i is " + str(i) + " : " + str(item.size()))
                    item_conv = item.permute(0,2,1)  # item: (a,b,c); item_conv: (a,c,b)
                    item_conv = self.convLst[i](item_conv)
                    item_conv = self.pooling(item_conv)
                    item_conv = item_conv.permute(0,2,1)  # item_conv: (a,b,c)
                    # print("From model.forward(): Size of item_conv when i is " + str(i) + " : " + str(item_conv.size()))
                    item = self.PosEncLst[i](item)
                    item_mha, _ = self.SEQLst[i](item, item, item)  # item: (a,b,c); item_mha: (a,b,c)
                    # print("From model.forward(): Size of item_mha when i is " + str(i) + " : " + str(item_mha.size()))
                    item = torch.cat((item_conv,item_mha[:, :item_conv.shape[1], :]),2)  # item_conv: (a,b,c); item_mha: (a,b,c); item: (a,b,c);
                    # item = torch.cat((item_conv,self.perform_dim_reductn(item_mha, item_conv.shape[1])),2)  # item_conv: (a,b,c); item_mha: (a,b,c); item: (a,b,c);
                    # print("From model.forward(): Size of item  at end when i is " + str(i) + " : " + str(item.size()))
                elif((i > 1) and (i <= self.numLayers-2)):  # intermediate layers between 2th and last layer : both conv and attention layers
                    # print("From model.forward(): Size of item at beginning when i is " + str(i) + " : " + str(item.size()))
                    item_conv = item.permute(0,2,1)  # item: (a,b,c); item_conv: (a,c,b)
                    item_conv = self.convLst[i](item_conv)
                    item_conv = self.pooling(item_conv)
                    item_conv = item_conv.permute(0,2,1)  # item_conv: (a,b,c)
                    # print("From model.forward(): Size of item_conv when i is " + str(i) + " : " + str(item_conv.size()))
                    item = self.PosEncLst[i](item)
                    item_mha, _ = self.SEQLst[i](item, item, item)  # item: (a,b,c); item_mha: (a,b,c)
                    # print("From model.forward(): Size of item_mha when i is " + str(i) + " : " + str(item_mha.size()))
                    item = torch.cat((item_conv,item_mha[:, :item_conv.shape[1], :]),2)  # item_conv: (a,b,c); item_mha: (a,b,c); item: (a,b,c);
                    # item = torch.cat((item_conv,self.perform_dim_reductn(item_mha, item_conv.shape[1])),2)  # item_conv: (a,b,c); item_mha: (a,b,c); item: (a,b,c);
                    # print("From model.forward(): Size of item  at end when i is " + str(i) + " : " + str(item.size()))
                elif(i == self.numLayers - 1):  # last layer: only conv layer and no attention/trx layer
                    # print("From model.forward(): Size of item at beginning when i is last layer: " + str(i) + " : " + str(item.size()))
                    item = item.permute(0,2,1)  # item: (a,c,b)
                    item = self.convLst[i](item)
                    # print("From model.forward(): Size of item (a,c,b)  at end when i is last layer: " + str(i) + " : " + str(item.size()))
                    item = item.mean(dim=2) #global average pooling over dim 2, reducing the data from 3D to 2D
                    # print("From model.forward(): Size of item after global average pooling over dim 2:" + " : " + str(item.size()))
                    protLst.append(item)


            # for i in range(0,self.numLayers-1):
            #     #conv1d and pooling expect hidden dim on 2nd axis (dim=1), gru needs hidden dim on 3rd axis (dim=2) . . .
            #     item = item.permute(0,2,1)
            #     item = self.convLst[i](item)
            #     item = self.pooling(item)
            #     item = item.permute(0,2,1)
            #     item2,hidden = self.GRULst[i](item)
            #     item = torch.cat((item,item2),2)
            
            # item = item.permute(0,2,1)
            # item = self.convLst[self.numLayers-1](item)
            # item = item.mean(dim=2) #global average pooling over dim 2, reducing the data from 3D to 2D
            # protLst.append(item)
        
        protA = protLst[0]
        protB = protLst[1]

        # now horizontally concatenate, auxProtA with protA and auxProtB with protB
        concat_protA = torch.cat((protA, auxProtA), 1)
        concat_protB = torch.cat((protB, auxProtB), 1)

        # x = torch.mul(concat_protA,concat_protB)  # element wise multiplication
        x = torch.cat((concat_protA, concat_protB), dim=1)  # side-by-side concatenation 
        
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.linear4(x)
        return x

    def perform_dim_reductn(self, X, n_comp):  # X.shape = (256, 666, 100); n_comp=221; Target X_transform.shape = (256, 221, 100)
        X = X.permute(0,2,1)  # X.shape = (256, 100, 666)
        # first, move X to CPU
        X = X.cpu()
        # then convert X from torch tensor to numpy array
        X = X.detach().numpy()
        X_transform = np.zeros((X.shape[0], X.shape[1], n_comp))
        X_2d = X.reshape(-1, X.shape[2])  # X_2d.shape = (256*100, 666)
        dim_reducer = PCA(n_components=n_comp, random_state=123)
        # dim_reducer = UMAP(n_components=n_comp, random_state=123)
        X_2d_transformed = dim_reducer.fit_transform(X_2d)  # X_2d_transformed.shape = (256*100, 221)
        X_transform = X_2d_transformed.reshape((X.shape[0], X.shape[1], n_comp))  # X_transform.shape = (256, 100, 221)
        # convert X_transform from numpy array to torch tensor
        X_transform = torch.tensor(X_transform, dtype=torch.float32)
        if self.fullGPU: #push everything to gpu
            X_transform = X_transform.to(torch.device(self.deviceType))
        # X_transform.shape = (256, 100, 221)
        X_transform = X_transform.permute(0,2,1)  # X_transform.shape = (256, 221, 100)
        return X_transform

    
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
        self.net = ChenNetwork(self.hiddenSize,self.inSize,self.aux_oneDencodingsize,self.numLayers,self.seed, self.fullGPU, self.deviceType)
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
        dataLookupSkip, dataMatrixSkip = self.loadEncodingFileWithPadding(featureFolder+'SkipGramAA7H5.encode',self.maxProteinLength)
        # dataLookupOneHot, dataMatrixOneHot = self.loadEncodingFileWithPadding(featureFolder+'OneHotEncoding7.encode',self.maxProteinLength)
        dataLookupLabelEncode, dataMatrixLabelEncode = self.loadLabelEncodingFileWithPadding(featureFolder+'LabelEncoding.encode',self.maxProteinLength)
        print("loading pssm_dict ...")
        # load the pssm values stored in pssm_dict
        pssm_dict_pkl_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data/benchmark_feat/PPI_Datasets/Human2021/', 'pssm_dict.pkl')
        pssm_dict = joblib.load(pssm_dict_pkl_path)
        # trimming pssm_dict so that it occupies less memory (RAM)
        for prot_id in list(pssm_dict.keys()):
            pssm_dict[prot_id]['seq'] = None
        print("loaded pssm_dict ...\n")
        print("loading blosum62_dict ...")
        # load the pssm values stored in blosum62_dict
        blosum62_dict_pkl_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data/benchmark_feat/PPI_Datasets/Human2021/', 'blosum62_dict.pkl')
        blosum62_dict = joblib.load(blosum62_dict_pkl_path)
        # trimming blosum62_dict so that it occupies less memory (RAM)
        for prot_id in list(blosum62_dict.keys()):
            blosum62_dict[prot_id]['seq'] = None
        print("loaded blosum62_dict ...\n")
        print("\nloading human_seq_feat_dict ...")
        # load tl-based 1d embeddings stored in human_seq_feat_dict
        human_seq_feat_dict_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data', 'human_seq_1d_feat_dict_prot_t5_xl_uniref50.pkl')
        human_seq_feat_dict = joblib.load(human_seq_feat_dict_path)
        # trimming human_seq_feat_dict so that it occupies less memory (RAM)
        for prot_id in list(human_seq_feat_dict.keys()):
            human_seq_feat_dict[prot_id]['seq'] = human_seq_feat_dict[prot_id]['seq_2d_feat'] = None
        print("loaded human_seq_feat_dict ...")

        # allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupOneHot.keys()))
        allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupLabelEncode.keys()))
        allProteinsList = list(human_seq_feat_dict.keys())

        # self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixOneHot.shape[1]
        self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixLabelEncode.shape[1] + \
                                           pssm_dict[str(allProteinsList[0])]['pssm_val'].shape[1] + blosum62_dict[str(allProteinsList[0])]['blosum62_val'].shape[1]
        self.aux_oneDencodingsize = len(human_seq_feat_dict[int(allProteinsList[0])]['seq_feat'])

        self.dataLookup = {}
        self.dataMatrix = torch.zeros((len(allProteinsSet),self.maxProteinLength,self.encodingSize))
        self.oneDdataMatrix = torch.zeros((len(allProteinsSet), self.aux_oneDencodingsize))

        for item in allProteinsSet:
            item = str(item)
            self.dataLookup[item] = len(self.dataLookup)
            skipData = dataMatrixSkip[dataLookupSkip[item],:,:].T
            # oneHotData = dataMatrixOneHot[dataLookupOneHot[item],:,:].T
            labelEncodeData = dataMatrixLabelEncode[dataLookupLabelEncode[item],:,:].T
            self.dataMatrix[self.dataLookup[item],:,:skipData.shape[1]] = skipData
            # self.dataMatrix[self.dataLookup[item],:,skipData.shape[1]:(skipData.shape[1] + oneHotData.shape[1])] = oneHotData
            self.dataMatrix[self.dataLookup[item],:,skipData.shape[1]:(skipData.shape[1] + labelEncodeData.shape[1])] = labelEncodeData

            # processing related to the current pssm-matrix - start
            cur_pssm_mat = pssm_dict[str(item)]['pssm_val']
            # # cur_pssm_mat = cur_pssm_mat /20.0  # normalize pssm values
            pssm_mat_nrows, pssm_mat_ncols = cur_pssm_mat.shape
            # if pssm_mat_nrows is greater than maxProteinLength, then chop the extra part
            if(pssm_mat_nrows > self.maxProteinLength):
                cur_pssm_mat = cur_pssm_mat[:self.maxProteinLength, :]
            # processing related to the current pssm-matrix - end
            self.dataMatrix[self.dataLookup[item],:cur_pssm_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1]):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols)] = cur_pssm_mat

            # processing related to the current blosum62-matrix - start
            cur_blosum62_mat = blosum62_dict[str(item)]['blosum62_val']
            # # cur_blosum62_mat = cur_blosum62_mat /20.0  # normalize blosum62 values
            blosum62_mat_nrows, blosum62_mat_ncols = cur_blosum62_mat.shape
            # if blosum62_mat_nrows is greater than maxProteinLength, then chop the extra part
            if(blosum62_mat_nrows > self.maxProteinLength):
                cur_blosum62_mat = cur_blosum62_mat[:self.maxProteinLength, :]
            # processing related to the current blosum62-matrix - end
            self.dataMatrix[self.dataLookup[item],:cur_blosum62_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols + blosum62_mat_ncols)] = cur_blosum62_mat

            self.oneDdataMatrix[self.dataLookup[item]] = torch.from_numpy(human_seq_feat_dict[int(item)]['seq_feat'])
        # end of for loop

        # perform the normalization of the auxiliary data matrix
        print('perform the normalization of the auxiliary data matrix')
        aux_data_arr = self.oneDdataMatrix.numpy()
        scaler = preprocessing.StandardScaler()
        aux_data_arr_scaled = scaler.fit_transform(aux_data_arr)
        self.oneDdataMatrix = torch.from_numpy(aux_data_arr_scaled)
