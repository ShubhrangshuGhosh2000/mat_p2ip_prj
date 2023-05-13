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
from proc.pipr_plus_AD.pipr_plus_origMan_auxTlOtherMan.piprp_ChenNetwork_origMan_auxTlOtherMan_rand50 import ChenNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.pipr_plus_AD.pipr_plus_origMan_auxTlOtherMan import piprp_RunTrainTest_origMan_auxTlOtherMan_rand50


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

#algorithms
li2020Test = True

#data Types
HumanRandom50 = True

# baseResultsFolderName = 'results/'
baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data_AD/benchmark_res/piprp_res/piprp_res_origMan_auxTlOtherMan_tune/')

#runs based on global variables
#can be toggled before calling function
def RunAll(findBestHparam=False):
    start_idx = end_idx = 0
    if li2020Test:
        #create results folders if they do not exist
        PPIPUtils.makeDir(baseResultsFolderName)
        # the following are used to find the overall best hparam combination
        overall_best_hparam_dict = {}
        best_score = 0.00

        hyp = {'fullGPU':True,'deviceType':'cuda'} 
        # hyp = {'fullGPU':False,'deviceType':'cpu'}

        # populate hyperparams
        hyperparameters = {}
        hyperparameters['hiddenSize'] = [40, 44, 50, 60, 70]
        hyperparameters['numLayers'] = [4, 5, 6]
        hyperparameters['leakyReLU_negSlope'] = [0.01, 0.03, 0.1, 0.3, 0.4]  # 0.3
        hyperparameters['n_heads'] = [1, 2]
        hyperparameters['layer_1_size'] = [1024, 800, 1600]  # 1024  # for the linear layers

        # hyperparameters['hiddenSize'] = [40, 45, 50]
        # hyperparameters['numLayers'] = [4, 5]
        # hyperparameters['leakyReLU_negSlope'] = [0.3]  # 0.3
        # hyperparameters['n_heads'] = [1]
        # hyperparameters['layer_1_size'] = [1024]  # 1024  # for the linear layers

        print("Creating grid of all possible hyper-parameters combinations...")
        hparam_grid_list = list(product(hyperparameters['hiddenSize'], hyperparameters['numLayers'], hyperparameters['leakyReLU_negSlope'] \
                                        , hyperparameters['n_heads'], hyperparameters['layer_1_size']))
        # iterate through the grid and perform the hyper-parameter tuning to find the best hyper-parameter combination
        print("Iterating through the grid and performing the hyper-parameter tuning to find the best hyper-parameter combination...")
        hparam_grid_list_len = len(hparam_grid_list)
        if(not findBestHparam):
            start_idx = 2*(hparam_grid_list_len//3)     # 0                          # hparam_grid_list_len//3           # 2*(hparam_grid_list_len//3)
            end_idx = hparam_grid_list_len              # hparam_grid_list_len//3    # 2*(hparam_grid_list_len//3)       # hparam_grid_list_len
        elif(findBestHparam):
            start_idx = 0
            end_idx = hparam_grid_list_len

        for itr in range(start_idx, end_idx):
            resultsFolderName= os.path.join(baseResultsFolderName, 'tune_' + str(itr))
            PPIPUtils.makeDir(resultsFolderName)
            hparam_combo_tuple = hparam_grid_list[itr]
            current_iter_hparam_combo_dict = {'hiddenSize': hparam_combo_tuple[0], 'numLayers': hparam_combo_tuple[1], 'leakyReLU_negSlope': hparam_combo_tuple[2],
                                              'n_heads': hparam_combo_tuple[3], 'layer_1_size': hparam_combo_tuple[4]}
            print("\n ### itr: " + str(itr) + ' out of '+ str(hparam_grid_list_len) + "\nhparam_combo: " + str(current_iter_hparam_combo_dict))

            hyp = {**hyp, **current_iter_hparam_combo_dict}
            outResultsName = os.path.join(resultsFolderName, 'piprp_res_origMan_auxTlOtherMan_tune_' + str(itr) + '.txt')
            if(not findBestHparam):
                trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName + str('/'))
                piprp_RunTrainTest_origMan_auxTlOtherMan_rand50.runTest(ChenNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=saves,predictionsFLst = pfs,startIdx=0)
                piprp_RunTrainTest_origMan_auxTlOtherMan_rand50.calcOverallScore_Pos50(outResultsName)
            elif(findBestHparam):
                outResultCsvFileName = outResultsName.replace('.txt', '.csv')
                outResultDf = pd.read_csv(outResultCsvFileName)
                crnt_score = outResultDf.at[0, 'avg_ACC']
                if(crnt_score > best_score):
                    best_score = crnt_score
                    overall_best_hparam_dict['best_itr_no'] = itr
                    overall_best_hparam_dict['best_hparam_combo_dict'] = hyp
                    overall_best_hparam_dict['best_score_dict'] = {'avg_ACC': crnt_score, 'avg_AUC': outResultDf.at[0, 'avg_AUC']}
        # end of for loop: for itr in range(start_idx, end_idx):
        if(not findBestHparam):
            print('\n ############## END OF TUNING: start_idx: ' + str(start_idx) + ' : end_idx: ' + str(end_idx)+ ' ################')
        elif(findBestHparam):
            # load the best tuned model
            best_model_FolderName= os.path.join(baseResultsFolderName, 'tune_' + str(overall_best_hparam_dict['best_itr_no']), 'Li2020_AD.out')
            best_state = torch.load(best_model_FolderName)
            overall_best_hparam_dict['best_lr'] = best_state['scheduler']['best']
            print('\n ############################ Best hyper-param combo related details -Start ##################\n')
            print(str(overall_best_hparam_dict))
            print('\n ############################ Best hyper-param combo related details -End ##################\n')


def genSequenceFeatures():
    featureDir = os.path.join(root_path, 'dataset/preproc_data_AD/benchmark_feat/')

    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Li_AD/',set(['AC30','LD10_CTD','PSAAC15','conjointTriad','PSEAAC_3','Random500','AllvsAllSim']))
    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['EGBW11','AC30','LD10_CTD','PSAAC15','conjointTriad','MMI','Moran','Geary','PSSMAAC','PSSMDPC','AC11','PSAAC9','SkipGramAA7','OneHotEncoding7','OneHotEncoding24','NumericEncoding22','AC14_30','MCD5CTD','SkipGramAA25H20','NumericEncoding20Skip3','Geary_Zhao_30','NMBroto_Zhao_30','Moran_Zhao_30','PSEAAC_Zhao_30','Grantham_Quasi_30','Schneider_Quasi_30','MCD4CTD','Grantham_Sequence_Order_30','Schneider_Sequence_Order_30','PSSMLST','SkipWeightedConjointTriad','PSAAC20','AAC20','AAC400','DUMULTIGROUPCTD','APSAAC30_2','JIA_DWT','MLD4CTD','NMBROTO_6_30','PSSMDCT','NMBROTO_9','MORAN_9','GEARY_9','PSEAAC_3','CHAOS','Random500','AllvsAllSim']))

    # extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['SkipGramAA7','LabelEncoding','PSSM', 'Blosum62'])) 


if __name__ == '__main__':
    # genSequenceFeatures()
    RunAll(findBestHparam=True)