import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mtf_p2ip_DS.mtf_p2ip_origMan_auxTlOtherMan.mtf_MtfP2ipNetwork_origMan_auxTlOtherMan_DS_test import MtfP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mtf_p2ip_DS.mtf_p2ip_origMan_auxTlOtherMan import mtf_RunTrainTest_origMan_auxTlOtherMan_DS


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mtf_p2ip_prj')


def execute(spec_type = 'human'): 
    print('\n########## spec_type: ' + str(spec_type))
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_DS/mtf_res_origMan_auxTlOtherMan_' + spec_type + '/')
    # create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['maxProteinLength'] = 800  # default: 50
    # # for normalization
    # hyp['featScaleClass'] = StandardScaler
    hyp['hiddenSize'] = 70  # default: 50
    hyp['numLayers'] = 4  # default: 6
    hyp['n_heads'] = 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers

    hyp['batchSize'] = 256  # default: 256  # for xai it is 5
    hyp['numEpochs'] = 10  # default: 100
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData(resultsFolderName, spec_type)
    outResultsName = os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_' + spec_type + '_DS.txt')
    # specifying human_full model location
    human_full_model_loc = os.path.join(root_path, 'dataset/proc_data_DS/mtf_res_origMan_auxTlOtherMan_human/DS_human_full.out')
    loads = [human_full_model_loc]
    mtf_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS(MtfP2ipNetworkModule, outResultsName,trainSets,testSets,featureFolder,hyp,resultsAppend=False,saveModels=None,predictionsFLst = pfs,startIdx=0,loads=loads,spec_type=spec_type)
    # mtf_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_DS_xai(MtfP2ipNetworkModule,testSets,featureFolder,hyp,startIdx=0,loads=loads,spec_type=spec_type,resultsFolderName=resultsFolderName)


def extract_prot_seq_feat(spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+spec_type+'/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']), spec_type = spec_type) 


if __name__ == '__main__':
    spec_type = 'ecoli'  # human, ecoli, fly, mouse, worm, yeast
    # extract_prot_seq_feat(spec_type)
    execute(spec_type)