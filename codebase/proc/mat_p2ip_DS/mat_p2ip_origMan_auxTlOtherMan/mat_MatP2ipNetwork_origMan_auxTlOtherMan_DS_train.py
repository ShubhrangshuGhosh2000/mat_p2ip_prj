import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
from utils.NetworkRunnerCollate import NetworkRunnerCollate
from proc.mat_p2ip_DS.GenericNetworkModel_mat import GenericNetworkModel
from proc.mat_p2ip_DS.GenericNetworkModule_mat import GenericNetworkModule
from proc.mat_p2ip_DS.positional_encoder_mat import PositionalEncoder
import torch
import torch.nn as nn
import joblib
import numpy as np


class MatP2ipNetwork(nn.Module):
    def __init__(self,hiddenSize=50,inSize=14,aux_oneDencodingsize=1024,numLayers=6,n_heads=2,layer_1_size=1024,seed=1,fullGPU=False,deviceType='cpu'):
        super(MatP2ipNetwork, self).__init__()
        torch.manual_seed(seed)
        self.pooling = nn.MaxPool1d(3)
        self.activation = nn.LeakyReLU(0.3)
        self.numLayers = numLayers
        self.fullGPU = fullGPU
        self.deviceType = deviceType

        
        self.n_heads = n_heads  
        dropout_pos_enc = 0.10
        n_encoder_layers = 1  
        

        self.convLst = nn.ModuleList()
        self.SEQLst = nn.ModuleList()
        self.PosEncLst = nn.ModuleList()
        for i in range(0,self.numLayers):
            if(i == 0):  
                self.convLst.append(nn.Conv1d(inSize,hiddenSize,3))  
                self.PosEncLst.append(None)  
                self.SEQLst.append(None)  
            elif(i == 1): 
                self.convLst.append(nn.Conv1d(hiddenSize,hiddenSize,3))
                self.PosEncLst.append(None)  
                self.SEQLst.append(nn.GRU(input_size=hiddenSize,hidden_size=hiddenSize,bidirectional=True,batch_first=True))
            elif((i > 1) and (i <= self.numLayers-2)):  
                self.convLst.append(nn.Conv1d(hiddenSize*(i+1),hiddenSize,3))
                self.PosEncLst.append(PositionalEncoder(d_model= hiddenSize*(i+1), dropout=dropout_pos_enc, max_seq_len=700, batch_first=True))
                self.SEQLst.append(nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=hiddenSize*(i+1), nhead=n_heads, dim_feedforward = hiddenSize, batch_first=True)
                                                        , num_layers=n_encoder_layers
                                                        , norm=None))
            elif(i == self.numLayers -1):  
                self.convLst.append(nn.Conv1d(hiddenSize*(i+1),hiddenSize,3))
                self.PosEncLst.append(None)  
                self.SEQLst.append(None)  

        self.otherManConvLst = nn.ModuleList()
        tl_1d_tensor_len = 1024  
        
        in_channels = aux_oneDencodingsize - tl_1d_tensor_len
        
        for i in range(0,self.numLayers):
            if i == 0: 
                self.otherManConvLst.append(nn.Conv1d(in_channels,hiddenSize*2,3))
            else:
                self.otherManConvLst.append(nn.Conv1d(hiddenSize*2,hiddenSize*2,3))

        self.layer_1_size = layer_1_size
        layer_2_size = layer_1_size//2
        layer_3_size =  layer_2_size//2
        self.linear1 = nn.Linear(2*(hiddenSize+tl_1d_tensor_len)+(2*hiddenSize), layer_1_size)  
        
        self.linear2 = nn.Linear(layer_1_size, layer_2_size)
        self.linear3 = nn.Linear(layer_2_size, layer_3_size)
        self.linear4 = nn.Linear(layer_3_size, 2)
        self.bn1 = nn.BatchNorm1d(layer_1_size)
        self.bn2 = nn.BatchNorm1d(layer_2_size)
        self.bn3 = nn.BatchNorm1d(layer_3_size)
        
    def forward(self,x):
        (protA, protB, auxProtA, auxProtB) = x
        protLst = []
        for item in [protA, protB]: 
            for i in range(0,self.numLayers):
                if(i == 0):   
                    item = item.permute(0,2,1)  
                    item = self.convLst[i](item)
                    item = self.pooling(item)
                    item = item.permute(0,2,1)  
                elif(i == 1):  
                    item_conv = item.permute(0,2,1)  
                    item_conv = self.convLst[i](item_conv)
                    item_conv = self.pooling(item_conv)
                    item_conv = item_conv.permute(0,2,1)  
                    item_gru,hidden = self.SEQLst[i](item)  
                    
                    item = torch.cat((item_conv,item_gru[:, :item_conv.shape[1], :]),2)  
                    
                elif((i > 1) and (i <= self.numLayers-2)):  
                    item_conv = item.permute(0,2,1)  
                    item_conv = self.convLst[i](item_conv)
                    item_conv = self.pooling(item_conv)
                    item_conv = item_conv.permute(0,2,1)  
                    
                    item = self.PosEncLst[i](item)
                    item_trx = self.SEQLst[i](item)  
                    
                    item = torch.cat((item_conv,item_trx[:, :item_conv.shape[1], :]),2)  
                    
                elif(i == self.numLayers - 1):  
                    item = item.permute(0,2,1)  
                    item = self.convLst[i](item)
                    
                    item = item.mean(dim=2) 
                    
                    protLst.append(item)
            
        protA = protLst[0]
        protB = protLst[1]

        batch_size = auxProtA.shape[0]
        tl_1d_tensor_len = 1024  
        aux_1d_tensor_len = auxProtA.shape[1]  
        tl_1d_auxProtA_tensor, other_man_1d_auxProtA_tensor = auxProtA.split([tl_1d_tensor_len, aux_1d_tensor_len - tl_1d_tensor_len], dim=1)
        tl_1d_auxProtB_tensor, other_man_1d_auxProtB_tensor = auxProtB.split([tl_1d_tensor_len, aux_1d_tensor_len - tl_1d_tensor_len], dim=1)
        other_man_1d_auxProtA_tensor_reshaped = other_man_1d_auxProtA_tensor.reshape(batch_size, -1, 1)
        other_man_1d_auxProtB_tensor_reshaped = other_man_1d_auxProtB_tensor.reshape(batch_size, 1, -1)
        other_man_2d_tensor = torch.matmul(other_man_1d_auxProtA_tensor_reshaped, other_man_1d_auxProtB_tensor_reshaped)
        
        for i in range(0,self.numLayers-1):
            other_man_2d_tensor = other_man_2d_tensor.permute(0,2,1)
            other_man_2d_tensor = self.otherManConvLst[i](other_man_2d_tensor)
            
            other_man_2d_tensor = self.pooling(other_man_2d_tensor)
            
            other_man_2d_tensor = other_man_2d_tensor.permute(0,2,1)
        other_man_2d_tensor = other_man_2d_tensor.permute(0,2,1)
        other_man_2d_tensor = self.otherManConvLst[self.numLayers-1](other_man_2d_tensor)
        
        other_man_1d_tensor = other_man_2d_tensor.mean(dim=2) 
        
        concat_protA = torch.cat((protA, tl_1d_auxProtA_tensor), dim=1)
        concat_protB = torch.cat((protB, tl_1d_auxProtB_tensor), dim=1)

        x = torch.cat((concat_protA, other_man_1d_tensor, concat_protB), dim=1)  

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

class NetworkRunnerMatP2ip(NetworkRunnerCollate):
    def __init__(self,net,batch_size=256,deviceType=None,lr=1e-2,optType='Adam',weight_decay=0,sched_factor=0.1,sched_patience=1,sched_cooldown=0,sched_thresh=1e-2,predictSoftmax=True,hyp={},skipScheduler=30):
        NetworkRunnerCollate.__init__(self,net,batch_size,deviceType,lr,optType,weight_decay,sched_factor,sched_patience,sched_cooldown,sched_thresh,predictSoftmax,hyp)
        self.skipScheduler = hyp.get('skipScheduler',skipScheduler)
    
            
    def updateScheduler(self,values):
        if self.scheduler is not None and self.epoch > self.skipScheduler:
            self.scheduler.step(values)

class MatP2ipModel(GenericNetworkModel):
    def __init__(self,hyp={},inSize=12,aux_oneDencodingsize=1024,hiddenSize=50,numLayers=6,fullGPU=False,deviceType=None,numEpochs=100,batchSize=256,lr=5e-4,minLr=1e-4,schedFactor=.5,schedPatience=3,schedThresh=1e-2,threshSchedMode='abs'):
        GenericNetworkModel.__init__(self,hyp=hyp,fullGPU=fullGPU,deviceType=deviceType,numEpochs=numEpochs,batchSize=batchSize,lr=lr,minLr=minLr,schedFactor=schedFactor,schedPatience=schedPatience,schedThresh=schedThresh)
        # ##### setting hyper-parameters
        self.inSize = inSize
        self.aux_oneDencodingsize = aux_oneDencodingsize
        self.hiddenSize = hyp.get('hiddenSize',hiddenSize)
        self.numLayers = hyp.get('numLayers',numLayers)
        self.n_heads = hyp.get('n_heads', 2)
        self.layer_1_size = hyp.get('layer_1_size', 1024)

        #move uncommon network runner properties into hyperparams list if needed
        hyp['amsgrad'] = hyp.get('amsgrad',True)
        hyp['threshSchedMode'] = hyp.get('threshSchedMode',threshSchedMode)
        
        
    def genModel(self):
        self.net = MatP2ipNetwork(self.hiddenSize,self.inSize,self.aux_oneDencodingsize,self.numLayers \
                                ,self.n_heads, self.layer_1_size, self.seed, self.fullGPU, self.deviceType)
        # ################################# for multi-gpu training -start ###############
        # MUST SET ENV VARIABLE 'CUDA_VISIBLE_DEVICES' in the runtime environment where multi-gpu training will take place:
        # export CUDA_VISIBLE_DEVICES=0,1
        # echo $CUDA_VISIBLE_DEVICES
        if self.fullGPU: # push everything to gpu
            self.net= nn.DataParallel(self.net)
            self.net.to(torch.device(self.deviceType))
        # ################################# for multi-gpu training -end ###############

        #self.model = NetworkRunnerCollate(self.net,hyp=self.hyp)
        # ################################# only for testing (when model is already given) -start ###############
        # self.skipScheduler = self.hyp.get('skipScheduler', 30)
        # ################################# only for testing (when model is already given) -end ###############
        self.model = NetworkRunnerMatP2ip(self.net,hyp=self.hyp,skipScheduler=self.skipScheduler)

    #train network
    def fit(self,pairLst,classes,dataMatrix,oneDdataMatrix,validationPairs=None, validationClasses=None):
        self.skipScheduler = 250000//classes.shape[0]
        super().fit(pairLst,classes,dataMatrix,oneDdataMatrix,validationPairs,validationClasses)


class MatP2ipNetworkModule(GenericNetworkModule):
    def __init__(self, hyperParams = {}, maxProteinLength=2000, hiddenSize=50,inSize=12, aux_oneDencodingsize=1024):
        GenericNetworkModule.__init__(self,hyperParams)
        self.maxProteinLength = self.hyperParams.get('maxProteinLength',maxProteinLength)
        self.inSize = self.hyperParams.get('inSize',inSize) #temporary value, until data is loaded in loadFeatureData function
        self.aux_oneDencodingsize = self.hyperParams.get('aux_oneDencodingsize',aux_oneDencodingsize) #temporary value, until data is loaded in loadFeatureData function
        self.hiddenSize = self.hyperParams.get('hiddenSize',hiddenSize)
        
    def genModel(self):
        self.model = MatP2ipModel(self.hyperParams,self.inSize,self.aux_oneDencodingsize,self.hiddenSize)


    def loadFeatureData_DS(self,featureFolder, spec_type=None):
        dataLookupSkip, dataMatrixSkip = self.loadEncodingFileWithPadding(featureFolder+'SkipGramAA7H5.encode',self.maxProteinLength)
        dataLookupLabelEncode, dataMatrixLabelEncode = self.loadLabelEncodingFileWithPadding(featureFolder+'LabelEncoding.encode',self.maxProteinLength)
        
        pssm_dict_pkl_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data_DS/derived_feat', spec_type, 'pssm_dict.pkl')
        pssm_dict = joblib.load(pssm_dict_pkl_path)
        
        for prot_id in list(pssm_dict.keys()):
            pssm_dict[prot_id]['seq'] = None
        
        blosum62_dict_pkl_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data_DS/derived_feat', spec_type, 'blosum62_dict.pkl')
        blosum62_dict = joblib.load(blosum62_dict_pkl_path)
        
        for prot_id in list(blosum62_dict.keys()):
            blosum62_dict[prot_id]['seq'] = None
        
        DS_seq_feat_dict_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data_DS', 'DS_seq_feat_dict_prot_t5_xl_uniref50_' + spec_type + '.pkl')
        DS_seq_feat_dict = joblib.load(DS_seq_feat_dict_path)
        
        for prot_id in list(DS_seq_feat_dict.keys()):
            DS_seq_feat_dict[prot_id]['seq'] = DS_seq_feat_dict[prot_id]['seq_2d_feat'] = None
        
        DS_seq_manual_feat_dict = joblib.load(os.path.join(Path(__file__).parents[5], 'dataset/preproc_data_DS','DS_seq_manual_feat_dict_' + spec_type + '.pkl'))
        
        for prot_id in list(DS_seq_manual_feat_dict.keys()):
            DS_seq_manual_feat_dict[prot_id]['seq'] = None
        

        allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupLabelEncode.keys()))
        allProteinsList = list(DS_seq_feat_dict.keys())

        self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixLabelEncode.shape[1] + \
                                           pssm_dict[str(allProteinsList[0])]['pssm_val'].shape[1] + blosum62_dict[str(allProteinsList[0])]['blosum62_val'].shape[1]
        self.aux_oneDencodingsize = 2242  # 2878  # at first run, it will fail but after observing 'aux_oneDencodingsize' print in the for loop below, its
                                          # proper value can be updated here, so that the rerun would be error free.
        self.dataLookup = {}
        self.dataMatrix = torch.zeros((len(allProteinsSet),self.maxProteinLength,self.encodingSize))
        self.oneDdataMatrix = torch.zeros((len(allProteinsSet), self.aux_oneDencodingsize))

        for item in allProteinsSet:
            item = str(item)
            self.dataLookup[item] = len(self.dataLookup)
            skipData = dataMatrixSkip[dataLookupSkip[item],:,:].T
            labelEncodeData = dataMatrixLabelEncode[dataLookupLabelEncode[item],:,:].T
            self.dataMatrix[self.dataLookup[item],:,:skipData.shape[1]] = skipData
            self.dataMatrix[self.dataLookup[item],:,skipData.shape[1]:(skipData.shape[1] + labelEncodeData.shape[1])] = labelEncodeData

            cur_pssm_mat = pssm_dict[str(item)]['pssm_val']
            pssm_mat_nrows, pssm_mat_ncols = cur_pssm_mat.shape
            
            if(pssm_mat_nrows > self.maxProteinLength):
                cur_pssm_mat = cur_pssm_mat[:self.maxProteinLength, :]
            
            self.dataMatrix[self.dataLookup[item],:cur_pssm_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1]):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols)] = cur_pssm_mat

            
            cur_blosum62_mat = blosum62_dict[str(item)]['blosum62_val']
            blosum62_mat_nrows, blosum62_mat_ncols = cur_blosum62_mat.shape
            
            if(blosum62_mat_nrows > self.maxProteinLength):
                cur_blosum62_mat = cur_blosum62_mat[:self.maxProteinLength, :]
            
            self.dataMatrix[self.dataLookup[item],:cur_blosum62_mat.shape[0], \
                            (skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols):(skipData.shape[1] + labelEncodeData.shape[1] + pssm_mat_ncols + blosum62_mat_ncols)] = cur_blosum62_mat

            
            tl_1d_embedd_tensor = torch.from_numpy(DS_seq_feat_dict[item]['seq_feat'])
            
            seq_manual_feat_dict = DS_seq_manual_feat_dict[item]['seq_manual_feat_dict']
            other_man_feat_lst = seq_manual_feat_dict['AC30'] + seq_manual_feat_dict['PSAAC15'] + seq_manual_feat_dict['ConjointTriad'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] + seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_D']
            other_man_feat_arr = np.array(other_man_feat_lst)
            other_man_1d_embedd_tensor = torch.from_numpy(other_man_feat_arr)
            
            aux_1d_tensor = torch.cat((tl_1d_embedd_tensor, other_man_1d_embedd_tensor))
            # print('aux_oneDencodingsize: ' + str(aux_1d_tensor.shape[0]))  # important print statement as its output will be used above
            self.oneDdataMatrix[self.dataLookup[item]] = aux_1d_tensor
        print('Inside the loadFeatureData_DS() method - End')
