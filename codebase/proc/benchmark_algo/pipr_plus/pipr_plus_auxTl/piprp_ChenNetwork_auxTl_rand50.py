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

from utils import dl_reproducible_result_util
from utils_benchmark.NetworkRunnerCollate import NetworkRunnerCollate
from proc.benchmark_algo.pipr_plus.GenericNetworkModel_piprp import GenericNetworkModel
from proc.benchmark_algo.pipr_plus.GenericNetworkModule_piprp import GenericNetworkModule
from proc.benchmark_algo.pipr_plus.positional_encoder_piprp import PositionalEncoder
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time


class ChenNetwork(nn.Module):
    def __init__(self,hiddenSize=50,inSize=14,aux_oneDencodingsize=1024,numLayers=6,n_heads=2,layer_1_size=1024,seed=1,fullGPU=False,deviceType='cpu'):
        super(ChenNetwork, self).__init__()
        torch.manual_seed(seed)
        self.pooling = nn.MaxPool1d(3)
        self.activation = nn.LeakyReLU(0.3)
        self.numLayers = numLayers
        self.fullGPU = fullGPU
        self.deviceType = deviceType
        tl_1d_tensor_len = 1024  # from 1d_ProtTrans_embeddings

        self.layer_1_size = layer_1_size  # 1024
        layer_2_size = layer_1_size//2  # 512
        layer_3_size =  layer_2_size//2  # 256
        self.linear1 = nn.Linear(2*tl_1d_tensor_len, layer_1_size)
        self.linear2 = nn.Linear(layer_1_size, layer_2_size)
        self.linear3 = nn.Linear(layer_2_size, layer_3_size)
        self.linear4 = nn.Linear(layer_3_size, 2)
        self.bn1 = nn.BatchNorm1d(layer_1_size)
        self.bn2 = nn.BatchNorm1d(layer_2_size)
        self.bn3 = nn.BatchNorm1d(layer_3_size)
        
    def forward(self,x):
        (protA, protB, auxProtA, auxProtB) = x

        tl_1d_tensor_len = 1024  # from 1d_ProtTrans_embeddings
        aux_1d_tensor_len = auxProtA.shape[1]  # auxProtA.shape => (batch_size, aux_1d_tensor_len)
        tl_1d_auxProtA_tensor, other_man_1d_auxProtA_tensor = auxProtA.split([tl_1d_tensor_len, aux_1d_tensor_len - tl_1d_tensor_len], dim=1)
        tl_1d_auxProtB_tensor, other_man_1d_auxProtB_tensor = auxProtB.split([tl_1d_tensor_len, aux_1d_tensor_len - tl_1d_tensor_len], dim=1)

        # x = torch.mul(concat_protA,concat_protB)  # element wise multiplication
        x = torch.cat((tl_1d_auxProtA_tensor, tl_1d_auxProtB_tensor), dim=1)  # side-by-side concatenation 

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
        self.net = ChenNetwork(self.hiddenSize,self.inSize,self.aux_oneDencodingsize,self.numLayers \
                                ,self.n_heads, self.layer_1_size, self.seed, self.fullGPU, self.deviceType)
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

        print('\n#### loading tl-based 1d embeddings ####')
        print("\nloading human_seq_feat_dict ...")
        # load tl-based 1d embeddings stored in human_seq_feat_dict
        human_seq_feat_dict_path = os.path.join(Path(__file__).parents[5], 'dataset/preproc_data', 'human_seq_1d_feat_dict_prot_t5_xl_uniref50.pkl')
        human_seq_feat_dict = joblib.load(human_seq_feat_dict_path)
        # trimming human_seq_feat_dict so that it occupies less memory (RAM)
        for prot_id in list(human_seq_feat_dict.keys()):
            human_seq_feat_dict[prot_id]['seq'] = human_seq_feat_dict[prot_id]['seq_2d_feat'] = None
        print("loaded human_seq_feat_dict ...")

        print('\n#### loading other manual 1d embeddings ####')
        print("loading human_seq_manual_feat_dict ...")
        human_seq_manual_feat_dict = joblib.load(os.path.join(Path(__file__).parents[5], 'dataset/preproc_data','human_seq_manual_feat_dict.pkl'))
        # trimming human_seq_manual_feat_dict so that it occupies less memory (RAM)
        for prot_id in list(human_seq_manual_feat_dict.keys()):
            human_seq_manual_feat_dict[prot_id]['seq'] = None
        print("loaded human_seq_manual_feat_dict ...")

        # allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupOneHot.keys()))
        allProteinsSet = set(list(dataLookupSkip.keys())) & set(list(dataLookupLabelEncode.keys()))
        allProteinsList = list(human_seq_feat_dict.keys())

        # self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixOneHot.shape[1]
        self.encodingSize = self.inSize = dataMatrixSkip.shape[1] + dataMatrixLabelEncode.shape[1] + \
                                           pssm_dict[str(allProteinsList[0])]['pssm_val'].shape[1] + blosum62_dict[str(allProteinsList[0])]['blosum62_val'].shape[1]
        # self.aux_oneDencodingsize = len(human_seq_feat_dict[int(allProteinsList[0])]['seq_feat'])
        self.aux_oneDencodingsize = 2242  # 2878  # at first run, it will fail but after observing 'aux_oneDencodingsize' print in the for loop below, its
                                          # proper value can be updated here, so that the rerun would be error free.

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

            # extract tl-based 1d embeddings
            tl_1d_embedd_tensor = torch.from_numpy(human_seq_feat_dict[int(item)]['seq_feat'])
            # extract other manual 1d embeddings 
            seq_manual_feat_dict = human_seq_manual_feat_dict[int(item)]['seq_manual_feat_dict']
            other_man_feat_lst = seq_manual_feat_dict['AC30'] + seq_manual_feat_dict['PSAAC15'] + seq_manual_feat_dict['ConjointTriad'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] + seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] \
                                # + seq_manual_feat_dict['CHAOS'] \
                                # + seq_manual_feat_dict['AAC20'] + seq_manual_feat_dict['AAC400'] \
                                # + seq_manual_feat_dict['Grantham_Sequence_Order_30'] + seq_manual_feat_dict['Schneider_Sequence_Order_30'] \
                                # + seq_manual_feat_dict['Grantham_Quasi_30'] + seq_manual_feat_dict['Schneider_Quasi_30'] + seq_manual_feat_dict['APSAAC30_2']
                                # + seq_manual_feat_dict['DuMultiCTD_C'] + seq_manual_feat_dict['DuMultiCTD_T'] + seq_manual_feat_dict['DuMultiCTD_D']
            other_man_feat_arr = np.array(other_man_feat_lst)
            other_man_1d_embedd_tensor = torch.from_numpy(other_man_feat_arr)
            # concat both tl_1d_embedd_tensor and other_man_1d_embedd_tensor
            aux_1d_tensor = torch.cat((tl_1d_embedd_tensor, other_man_1d_embedd_tensor))
            # print('aux_oneDencodingsize: ' + str(aux_1d_tensor.shape[0]))  # important print statement as its output will be used above
            self.oneDdataMatrix[self.dataLookup[item]] = aux_1d_tensor
        # end of for loop

        # # # perform the full normalization of the auxiliary data matrix
        # # print('perform the full normalization of the auxiliary data matrix')
        # # aux_data_arr = self.oneDdataMatrix.numpy()
        # # scaler = preprocessing.StandardScaler()
        # # aux_data_arr_scaled = scaler.fit_transform(aux_data_arr)
        # # self.oneDdataMatrix = torch.from_numpy(aux_data_arr_scaled)

        # # perform the partial normalization (only tl part) of the auxiliary data matrix
        # print('perform the partial normalization (only tl part) of the auxiliary data matrix')
        # aux_data_arr = self.oneDdataMatrix.numpy()
        # aux_tl_1d_data_arr = aux_data_arr[:, : tl_1d_embedd_tensor.shape[0]]
        # aux_otherMan_1d_data_arr = aux_data_arr[:, tl_1d_embedd_tensor.shape[0]:]
        # scaler = preprocessing.StandardScaler()
        # aux_tl_1d_data_arr_scaled = scaler.fit_transform(aux_tl_1d_data_arr)
        # aux_data_arr_scaled = np.concatenate((aux_tl_1d_data_arr_scaled, aux_otherMan_1d_data_arr), axis=1)
        # self.oneDdataMatrix = torch.from_numpy(aux_data_arr_scaled)
        print('End of loadFeatureData() method')
