import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_DS_test import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_DS


root_path = os.path.join('/project/root/directory/path/here')



def execute(spec_type = 'human'): 
    
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_DS/mat_res_origMan_auxTlOtherMan_' + spec_type + '/')
    
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['maxProteinLength'] = 800  
    
    # hyp['featScaleClass'] = StandardScaler
    hyp['hiddenSize'] = 70  
    hyp['numLayers'] = 4  
    hyp['n_heads'] = 1  
    hyp['layer_1_size'] = 1024  

    hyp['batchSize'] = 256  
    hyp['numEpochs'] = 10  

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData(resultsFolderName, spec_type)
    outResultsName = os.path.join(resultsFolderName, 'mat_res_origMan_auxTlOtherMan_' + spec_type + '_DS.txt')
    # specifying human_full model location
    human_full_model_loc = os.path.join(root_path, 'dataset/proc_data_DS/mat_res_origMan_auxTlOtherMan_human/DS_human_full.out')
    loads = [human_full_model_loc]
    mat_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS(MatP2ipNetworkModule, outResultsName,trainSets,testSets,featureFolder,hyp,resultsAppend=False,saveModels=None,predictionsFLst = pfs,startIdx=0,loads=loads,spec_type=spec_type)
    # mat_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS_xai(MatP2ipNetworkModule,testSets,featureFolder,hyp,startIdx=0,loads=loads,spec_type=spec_type,resultsFolderName=resultsFolderName)


def extract_prot_seq_feat(spec_type = 'human'):
    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+spec_type+'/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']), spec_type = spec_type) 


if __name__ == '__main__':
    spec_type = 'ecoli'  # human, ecoli, fly, mouse, worm, yeast
    # extract_prot_seq_feat(spec_type)
    execute(spec_type)