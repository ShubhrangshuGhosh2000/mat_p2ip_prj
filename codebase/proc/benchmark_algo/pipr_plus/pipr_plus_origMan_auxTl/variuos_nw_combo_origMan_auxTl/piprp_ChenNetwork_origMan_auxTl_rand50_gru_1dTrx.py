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
from sklearn import preprocessing
from fast_transformers.builders import TransformerEncoderBuilder


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
            if i<= (self.numLayers - 2): # 4: #only numlayers-1 grus
                self.GRULst.append(nn.GRU(input_size=hiddenSize,hidden_size=hiddenSize,bidirectional=True,batch_first=True))

        # ############################ Transformer encoder part -start ############################
        n_heads = 6  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number. Default: 8
        d_model = 360  # This can be any value divisible by n_heads. 512 is used in the original transformer paper. Default 512
        d_model = d_model - (d_model % n_heads)  # to make d_model divisible by n_heads
        self.n_features = 1
        seq_length = 2 * hiddenSize
        dropout_pos_enc = 0.10
        n_encoder_layers = 4  # Number of times the encoder layer is stacked in the encoder. Default: 4

        # The encoder input layer is simply implemented as an nn.Linear() layer. The in_features argument must be equal 
        # to the number of variables you’re using as input to the model. In a univariate time series forecasting problem, in_features = 1. 
        # The out_features argument must be d_model which is a hyperparameter that has 
        # the value 512 in [“Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case”].
        self.encoder_input_layer = nn.Linear(in_features=self.n_features
                                            , out_features=d_model)
        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(d_model=d_model
                                                              , dropout=dropout_pos_enc
                                                              , max_seq_len=seq_length
                                                              , batch_first=True)

        # Create an encoder layer#
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model
                                                   , nhead=n_heads
                                                   , batch_first=True)

        # Stack the encoder layer n times in nn.TransformerEncoder
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer
                                            , num_layers=n_encoder_layers 
                                            , norm=None)
        # ############################ Transformer encoder part -end ############################

        # # ############################ Fast-Transformer encoder part -start ############################
        # n_heads = 4
        # self.n_features = 1
        # seq_length = 2*(hiddenSize+aux_oneDencodingsize)
        # d_model = 64
        # d_model = d_model - (d_model % n_heads)  # to make d_model divisible by n_heads
        # dropout_pos_enc = 0.1
        # n_encoder_layers = 2

        # # The encoder input layer is simply implemented as an nn.Linear() layer. The in_features argument must be equal 
        # # to the number of variables you’re using as input to the model. In a univariate time series forecasting problem, in_features = 1. 
        # # The out_features argument must be d_model which is a hyperparameter that has 
        # # the value 512 in [“Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case”].
        # self.encoder_input_layer = nn.Linear(in_features=self.n_features, out_features=d_model)
        # # # Create a list of fast_encoder layers
        # # fast_encoder_layers_lst = []
        # # for i in range(0,n_encoder_layers):
        # #     # if i == 0:
        # #     fast_encoder_layer = TransformerEncoderLayer(attention='ImprovedClusteredAttention'
        # #                                            , d_model=d_model
        # #                                            , d_ff = d_model  # default: d_model*4
        # #                                            , dropout=dropout_pos_enc
        # #                                            , activation='gelu'
        # #                                            )
        # #     fast_encoder_layers_lst.append(fast_encoder_layer)

        # # # Stack the encoder layer n times in TransformerEncoder
        # # self.fast_encoder = TransformerEncoder(layers=fast_encoder_layers_lst, norm_layer=None)

        # # Build a transformer encoder
        # self.fast_encoder = TransformerEncoderBuilder.from_kwargs(
        #     n_layers=n_encoder_layers,  # The number of transformer layers.
        #     n_heads=n_heads,
        #     query_dimensions=d_model // n_heads,  # d_model = query_dimensions * n_heads
        #     value_dimensions=d_model // n_heads,
        #     feed_forward_dimensions=1024,  # The dimensions of the fully connected layer in the transformer layers.
        #     attention_type="improved-clustered", # change this to use another attention implementation
        #     # attention_type="full", # change this to use another attention implementation
        #     activation="gelu",
        #     topk=32,  # used by improved clustered
        #     clusters=256,  # used by improved clustered
        #     bits=32,  # used by improved clustered
        # ).get()

        # # ############################ Fast-Transformer encoder part -end ############################

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

        x = torch.cat((protA, protB), dim=1)  # side-by-side concatenation 

        # ############################ Transformer encoder part -start ############################
        # Pass throguh the input layer right before the encoder
        x = x.reshape(x.shape[0], x.shape[1], self.n_features)  # convert shape of x as (batch_size, x length, n_features) 
        x = self.encoder_input_layer(x) # x shape: [batch_size, x length, d_model] regardless of number of input features
        # print("From model.forward(): Size of x after input layer: {}".format(x.size()))

        # Pass through the positional encoding layer
        x = self.positional_encoding_layer(x) # x shape: [batch_size, x length, d_model] regardless of number of input features
        # print("From model.forward(): Size of x after pos_enc layer: {}".format(x.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this use case, because all the
        # input sequences are naturally of the same length. 
        # (https://github.com/huggingface/transformers/issues/4083)
        x = self.encoder(x) # x shape: [batch_size, enc_seq_len, d_model]
        # print("From model.forward(): Size of x after encoder: {}".format(x.size()))
        # ############################ Transformer encoder part -end ############################

        # # ############################ Fast-Transformer encoder part -start ############################
        # # Pass throguh the input layer right before the encoder
        # x = x.reshape(x.shape[0], x.shape[1], self.n_features)  # convert shape of x as (batch_size, x length, n_features) 
        # x = self.encoder_input_layer(x) # x shape: [batch_size, x length, d_model] regardless of number of input features
        # # print("From model.forward(): Size of x after input layer: {}".format(x.size()))

        # # Pass through all the stacked encoder layers in the encoder
        # # Masking is only needed in the encoder if input sequences are padded
        # # which they are not in this use case, because all the
        # # input sequences are naturally of the same length. 
        # # (https://github.com/huggingface/transformers/issues/4083)
        # x = self.fast_encoder(x) # x shape: [batch_size, enc_seq_len, d_model]
        # # print("From model.forward(): Size of x after encoder: {}".format(x.size()))

        # # pass the encoder output to the linear layer
        # # ############################ Fast-Transformer encoder part -end ############################
        x = x.mean(dim=2) #global average pooling over dim 2, reducing the data from 3D to 2D
        # print("From model.forward(): Size of x after global average pooling over dim 2: {}".format(x.size()))

        # # now horizontally concatenate, auxProtA with protA and auxProtB with protB
        # concat_protA = torch.cat((protA, auxProtA), 1)
        # concat_protB = torch.cat((protB, auxProtB), 1)

        # # x = torch.mul(concat_protA,concat_protB)  # element wise multiplication
        # x = torch.cat((concat_protA, concat_protB), dim=1)  # side-by-side concatenation 
        x = torch.cat((auxProtA, x, auxProtB), dim=1)

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
        # print("From model.forward(): Size of x after linear4: {}".format(x.size()))

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
