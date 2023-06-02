import sys, os
from itertools import product
import pandas as pd
import torch
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mat_p2ip_AD.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_rand50 import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mat_p2ip_AD.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_rand50


root_path = os.path.join('/project/root/directory/path/here')


baseResultsFolderName = os.path.join(root_path, 'dataset/proc_data_AD/mat_res/mat_res_origMan_auxTlOtherMan_tune/')

def execute(findBestHparam=False):
    start_idx = end_idx = 0
    
    PPIPUtils.makeDir(baseResultsFolderName)
    
    overall_best_hparam_dict = {}
    best_score = 0.00

    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    
    hyperparameters = {}
    hyperparameters['hiddenSize'] = [40, 44, 50, 60, 70]
    hyperparameters['numLayers'] = [4, 5, 6]
    hyperparameters['leakyReLU_negSlope'] = [0.01, 0.03, 0.1, 0.3, 0.4]  # 0.3
    hyperparameters['n_heads'] = [1, 2]
    hyperparameters['layer_1_size'] = [1024, 800, 1600]  

    
    hparam_grid_list = list(product(hyperparameters['hiddenSize'], hyperparameters['numLayers'], hyperparameters['leakyReLU_negSlope'] \
                                    , hyperparameters['n_heads'], hyperparameters['layer_1_size']))
    
    
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
        outResultsName = os.path.join(resultsFolderName, 'mat_res_origMan_auxTlOtherMan_tune_' + str(itr) + '.txt')
        if(not findBestHparam):
            trainSets, testSets, saves, pfs, folderName = loadLiADData(resultsFolderName + str('/'))
            mat_RunTrainTest_origMan_auxTlOtherMan_rand50.runTest(MatP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=saves,predictionsFLst = pfs,startIdx=0)
            mat_RunTrainTest_origMan_auxTlOtherMan_rand50.calcOverallScore_Pos50(outResultsName)
        elif(findBestHparam):
            outResultCsvFileName = outResultsName.replace('.txt', '.csv')
            outResultDf = pd.read_csv(outResultCsvFileName)
            crnt_score = outResultDf.at[0, 'avg_ACC']
            if(crnt_score > best_score):
                best_score = crnt_score
                overall_best_hparam_dict['best_itr_no'] = itr
                overall_best_hparam_dict['best_hparam_combo_dict'] = hyp
                overall_best_hparam_dict['best_score_dict'] = {'avg_ACC': crnt_score, 'avg_AUC': outResultDf.at[0, 'avg_AUC']}
    
    if(not findBestHparam):
        print('\n ############## END OF TUNING: start_idx: ' + str(start_idx) + ' : end_idx: ' + str(end_idx)+ ' ################')
        pass
    elif(findBestHparam):
        
        best_model_FolderName= os.path.join(baseResultsFolderName, 'tune_' + str(overall_best_hparam_dict['best_itr_no']), 'Li2020_AD.out')
        best_state = torch.load(best_model_FolderName)
        overall_best_hparam_dict['best_lr'] = best_state['scheduler']['best']
        print(str(overall_best_hparam_dict))


def extract_prot_seq_feat():
    featureDir = os.path.join(root_path, 'dataset/preproc_data_AD/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Li_AD/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7'])) 


if __name__ == '__main__':
    # extract_prot_seq_feat()
    execute(findBestHparam=True)