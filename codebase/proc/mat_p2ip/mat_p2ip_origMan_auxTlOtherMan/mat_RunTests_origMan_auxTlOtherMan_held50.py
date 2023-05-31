import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_held50 import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_held50


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

resultsFolderName = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan/')

def execute():
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['hiddenSize'] = 70  
    hyp['numLayers'] = 4  
    hyp['n_heads'] = 1  
    hyp['layer_1_size'] = 1024  
    

    trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
    outResultsName = os.path.join(resultsFolderName, 'mat_res_origMan_auxTlOtherMan_held50.txt')
    
    mat_RunTrainTest_origMan_auxTlOtherMan_held50.runTest(MatP2ipNetworkModule, outResultsName,trainSets,testSets,folderName,hyp,resultsAppend=True,saveModels=saves,predictionsFLst = pfs,startIdx=0)
    mat_RunTrainTest_origMan_auxTlOtherMan_held50.calcOverallScore_Pos50(outResultsName)


def extract_prot_seq_feat():
    featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7'])) 

if __name__ == '__main__':
    # extract_prot_seq_feat()
    execute()