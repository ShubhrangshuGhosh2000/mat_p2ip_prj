import sys, os
import pandas as pd

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
# from Methods.Li2020DeepEnsemble.LiDeepNetwork import LiDeepNetworkModule
from proc.benchmark_algo.li20.li20_man_orig.LiDeepNetwork_man_orig_rand50 import LiDeepNetworkModule
import time
from preproc.benchmark_preproc.ProjectDataLoader import *
from preproc.benchmark_preproc.PreProcessDatasets import createFeatures
from proc.benchmark_algo.li20.li20_man_orig import RunTrainTest_man_orig_Rand50


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

#algorithms
li2020Test = True

#data Types
orgData = False
HumanRandom50 = True
HumanRandom20 = False
HumanHeldOut50 = False
HumanHeldOut20 = False

# baseResultsFolderName = 'results/'
baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data/benchmark_res/li20_res/li20_res_man_orig/')


#runs based on global variables
#can be toggled before calling function
def RunAll():    
    if li2020Test:
        #create results folders if they do not exist
        PPIPUtils.makeDir(baseResultsFolderName)
        # resultsFolderName=  baseResultsFolderName+'Li2020Results/'
        resultsFolderName=  baseResultsFolderName
        PPIPUtils.makeDir(resultsFolderName)
        hyp = {'fullGPU':True,'deviceType':'cuda'} 
        # hyp = {'fullGPU':False,'deviceType':'cpu'}

        # if orgData:
        #     outResultsName = os.path.join(resultsFolderName, 'li20_res_man_orig_OrigData.txt')
        #     trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
        #     runTest(LiDeepNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)

        if HumanRandom50:
            trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
            outResultsName = os.path.join(resultsFolderName, 'li20_res_man_orig_Rand50.txt')
            # RunTrainTest_man_orig_Rand50.runTest(LiDeepNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs,startIdx=0)
            RunTrainTest_man_orig_Rand50.runTest(LiDeepNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=None,predictionsFLst = pfs,startIdx=0)
            RunTrainTest_man_orig_Rand50.calcOverallScore_Pos50(outResultsName)

        # if HumanRandom20:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
        #     outResultsNameLst = []
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'li20_res_man_orig_Rand20_1.txt'))
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'li20_res_man_orig_Rand20_2.txt'))
        #     runTestLst(LiDeepNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True, modelsLst=None,saveModels=convertToFolder(saves),predictionsFLst = pfs)
        #     calcOverallScore_Pos20(outResultsNameLst)

        # if HumanHeldOut50:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
        #     outResultsName = os.path.join(resultsFolderName, 'li20_res_man_orig_Held50.txt')
        #     runTest(LiDeepNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)
        #     calcOverallScore_Pos50(outResultsName)

        # if HumanHeldOut20:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
        #     outResultsNameLst = []
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'li20_res_man_orig_Held20_1.txt'))
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'li20_res_man_orig_Held20_2.txt'))
        #     runTestLst(LiDeepNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)
        #     calcOverallScore_Pos20(outResultsNameLst)


def genSequenceFeatures():
    featureDir = os.path.join(root_path, 'dataset/preproc_data/benchmark_feat/')

    # createFeatures(featureDir+'PPI_Datasets/Li_AD/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
    # createFeatures(featureDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))

    # createFeatures(featureDir+'PPI_Datasets/Human2021/',set(['AC30', 'LD10_CTD', 'PSAAC15','conjointTriad'])) 

if __name__ == '__main__':
    genSequenceFeatures()
    RunAll()