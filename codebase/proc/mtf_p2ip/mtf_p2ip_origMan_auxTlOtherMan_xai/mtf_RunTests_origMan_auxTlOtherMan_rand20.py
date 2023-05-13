import sys, os
from sklearn.preprocessing import StandardScaler


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
from proc.mtf_p2ip.mtf_p2ip_origMan_auxTlOtherMan_xai.mtf_MtfP2ipNetwork_origMan_auxTlOtherMan_rand20 import MtfP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mtf_p2ip.mtf_p2ip_origMan_auxTlOtherMan_xai import mtf_RunTrainTest_origMan_auxTlOtherMan_rand20


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mtf_p2ip_prj')

#algorithms
mtfP2ip2019RNN = True

#data Types
orgData = False
HumanRandom50 = False
HumanRandom20 = True
HumanHeldOut50 = False
HumanHeldOut20 = False

# baseResultsFolderName = 'results/'
baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data/mtf_res/mtf_res_origMan_auxTlOtherMan_xai/')


#runs based on global variables
#can be toggled before calling function
def RunAll():    
    if mtfP2ip2019RNN:
        #create results folders if they do not exist
        PPIPUtils.makeDir(baseResultsFolderName)
        # resultsFolderName=  baseResultsFolderName+'mtf20Results/'
        resultsFolderName=  baseResultsFolderName
        PPIPUtils.makeDir(resultsFolderName)
        hyp = {'fullGPU':True,'deviceType':'cuda'} 
        # hyp = {'fullGPU':False,'deviceType':'cpu'}

        # # for normalization
        # hyp['featScaleClass'] = StandardScaler
        hyp['hiddenSize'] = 70  # default: 50
        hyp['numLayers'] = 4  # default: 6
        # ## hyp['leakyReLU_negSlope'] = [0.01, 0.03, 0.1, 0.3, 0.4]  # default: 0.3
        hyp['n_heads'] = 1  # default: 2
        hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers

        hyp['batchSize'] = 5  # default: 256  # for xai it is 5
        hyp['numEpochs'] = 100  # default: 100
        print('hyp: ' + str(hyp))

        # if orgData:
        #     outResultsName = os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_OrigData.txt')
        #     trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName)
        #     runTest(MtfP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)

        # if HumanRandom50:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanRandom50(resultsFolderName)
        #     outResultsName = os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_Rand50.txt')
        #     # mtf_RunTrainTest_origMan_auxTlOtherMan_Rand50.runTest(MtfP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs,startIdx=0)
        #     mtf_RunTrainTest_origMan_auxTlOtherMan_rand50.runTest(MtfP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=None,predictionsFLst = pfs,startIdx=0)
        #     mtf_RunTrainTest_origMan_auxTlOtherMan_rand50.calcOverallScore_Pos50(outResultsName)

        if HumanRandom20:
            trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
            outResultsNameLst = []
            outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_rand20_1.txt'))
            outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_rand20_2.txt'))
            # mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.runTestLst(MtfP2ipNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True, modelsLst=None,saveModels=saves,predictionsFLst = pfs,startIdx=0)
            # mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.calcOverallScore_Pos20(outResultsNameLst)
            mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.runTestLst_xai(MtfP2ipNetworkModule,testSets,folderName,hyp,predictionsFLst=pfs,startIdx=2,loads=saves)


        # if HumanHeldOut50:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
        #     outResultsName = os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_Held50.txt')
        #     runTest(MtfP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)
        #     calcOverallScore_Pos50(outResultsName)

        # if HumanHeldOut20:
        #     trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut20(resultsFolderName)
        #     outResultsNameLst = []
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_Held20_1.txt'))
        #     outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_Held20_2.txt'))
        #     runTestLst(MtfP2ipNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=convertToFolder(saves),predictionsFLst = pfs)
        #     calcOverallScore_Pos20(outResultsNameLst)


def genSequenceFeatures():
    featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')

    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Li_AD/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))

    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['SkipGramAA7','LabelEncoding','PSSM', 'Blosum62'])) 

if __name__ == '__main__':
    # genSequenceFeatures()
    RunAll()