import sys, os
from sklearn.preprocessing import StandardScaler
from itertools import product
import pandas as pd
import torch


# # currentdir = os.path.dirname(os.path.realpath(__file__))
# # parentdir = os.path.dirname(currentdir)
# # sys.path.append(parentdir)
# # currentDir = currentdir + '/'

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
# import PPIPUtils
from utils import PPIPUtils
from proc.mtf_p2ip_DS.mtf_p2ip_origMan_auxTlOtherMan.mtf_MtfP2ipNetwork_origMan_auxTlOtherMan_DS_train import MtfP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mtf_p2ip_DS.mtf_p2ip_origMan_auxTlOtherMan import mtf_RunTrainTest_origMan_auxTlOtherMan_DS


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')


# runs based on global variables
# can be toggled before calling function
def RunAll(spec_type = 'human'): 
    print('\n########## spec_type: ' + str(spec_type))

    baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data_DS/mtf_res_origMan_auxTlOtherMan_' + spec_type + '/')
    #create results folders if they do not exist
    PPIPUtils.makeDir(baseResultsFolderName)
    # resultsFolderName=  baseResultsFolderName+'mtf20Results/'
    resultsFolderName=  baseResultsFolderName
    PPIPUtils.makeDir(resultsFolderName)

    # MUST SET ENV VARIABLE 'CUDA_VISIBLE_DEVICES' in the runtime environment where multigpu training will take place:
    # export CUDA_VISIBLE_DEVICES=0,1
    # echo $CUDA_VISIBLE_DEVICES
    hyp = {'fullGPU':True, 'deviceType':'cuda'} 
    # hyp = {'fullGPU':False,'deviceType':'cpu'}

    hyp['maxProteinLength'] = 800  # default: 2000
    # # for normalization
    # hyp['featScaleClass'] = StandardScaler
    hyp['hiddenSize'] = 70  # 70  # default: 50
    hyp['numLayers'] = 4 # 4  # default: 6
    # ## hyp['leakyReLU_negSlope'] = [0.01, 0.03, 0.1, 0.3, 0.4]  # default: 0.3
    hyp['n_heads'] = 1 # 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers

    hyp['batchSize'] = 256  # default: 256
    hyp['numEpochs'] = 150  # default: 100
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData_human_full(resultsFolderName)
    mtf_RunTrainTest_origMan_auxTlOtherMan_DS.runTrainOnly_DS(MtfP2ipNetworkModule, trainSets, featureFolder, hyp, saves, spec_type)


def genSequenceFeatures(spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/benchmark_feat/')

    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Li_DS/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))

    # extract_prot_seq_2D_manual_feat(featureDir+spec_type+'/',set(['SkipGramAA7','LabelEncoding','PSSM', 'Blosum62']), spec_type = spec_type) 

if __name__ == '__main__':
    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast
    # genSequenceFeatures(spec_type)
    RunAll(spec_type)