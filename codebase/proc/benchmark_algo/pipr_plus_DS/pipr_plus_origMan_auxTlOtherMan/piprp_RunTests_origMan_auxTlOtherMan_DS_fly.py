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
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
# import PPIPUtils
from utils_benchmark import PPIPUtils
from proc.benchmark_algo.pipr_plus_DS.pipr_plus_origMan_auxTlOtherMan.piprp_ChenNetwork_origMan_auxTlOtherMan_DS_test import ChenNetworkModule
from preproc.benchmark_preproc.ProjectDataLoader import *
from preproc.benchmark_preproc.PreProcessDatasets import createFeatures
from proc.benchmark_algo.pipr_plus_DS.pipr_plus_origMan_auxTlOtherMan import piprp_RunTrainTest_origMan_auxTlOtherMan_DS


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')


# runs based on global variables
# can be toggled before calling function
def RunAll(spec_type = 'human'): 
    print('\n########## spec_type: ' + str(spec_type))

    baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data_DS/piprp_res_origMan_auxTlOtherMan_' + spec_type + '/')
    #create results folders if they do not exist
    PPIPUtils.makeDir(baseResultsFolderName)
    # resultsFolderName=  baseResultsFolderName+'piprp20Results/'
    resultsFolderName=baseResultsFolderName
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 
    # hyp = {'fullGPU':False,'deviceType':'cpu'}

    hyp['maxProteinLength'] = 800  # default: 50
    # # for normalization
    # hyp['featScaleClass'] = StandardScaler
    hyp['hiddenSize'] = 70  # default: 50
    hyp['numLayers'] = 4  # default: 6
    # ## hyp['leakyReLU_negSlope'] = [0.01, 0.03, 0.1, 0.3, 0.4]  # default: 0.3
    hyp['n_heads'] = 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers

    hyp['batchSize'] = 256  # default: 256  # for xai it is 5
    hyp['numEpochs'] = 10  # default: 100
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData(resultsFolderName, spec_type)
    outResultsName = os.path.join(resultsFolderName, 'piprp_res_origMan_auxTlOtherMan_' + spec_type + '_DS.txt')
    # specifying human_full model location
    human_full_model_loc = os.path.join(root_path, 'dataset/proc_data_DS/piprp_res_origMan_auxTlOtherMan_human/DS_human_full.out')
    loads = [human_full_model_loc]
    piprp_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS(ChenNetworkModule, outResultsName,trainSets,testSets,featureFolder,hyp,resultsAppend=False,saveModels=None,predictionsFLst = pfs,startIdx=0,loads=loads,spec_type=spec_type)
    # piprp_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS_xai(ChenNetworkModule,testSets,featureFolder,hyp,startIdx=0,loads=loads,spec_type=spec_type,resultsFolderName=resultsFolderName)


def genSequenceFeatures(spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/benchmark_feat/')

    # createFeatures(featureDir+'PPI_Datasets/Li_DS/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
    # createFeatures(featureDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))

    # createFeatures(featureDir+'PPI_Datasets/Human2021/',set(['SkipGramAA7','LabelEncoding','PSSM', 'Blosum62'])) 
    createFeatures(featureDir+spec_type+'/',set(['SkipGramAA7','LabelEncoding','PSSM', 'Blosum62']), spec_type = spec_type) 


if __name__ == '__main__':
    spec_type = 'fly'  # human, ecoli, fly, mouse, worm, yeast
    # genSequenceFeatures(spec_type)
    RunAll(spec_type)