import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan_xai.mat_MatP2ipNetwork_origMan_auxTlOtherMan_held50 import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mat_p2ip.mat_p2ip_origMan_auxTlOtherMan_xai import mat_RunTrainTest_origMan_auxTlOtherMan_held50


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

resultsFolderName = os.path.join(root_path, 'dataset/proc_data/mat_res/mat_res_origMan_auxTlOtherMan_xai/')

def execute():
    #create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['hiddenSize'] = 70  # default: 50
    hyp['numLayers'] = 4  # default: 6
    hyp['n_heads'] = 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, folderName = loadHumanHeldOut50(resultsFolderName)
    mat_RunTrainTest_origMan_auxTlOtherMan_held50.runTest_xai(MatP2ipNetworkModule,testSets,folderName,hyp,predictionsFLst=pfs,startIdx=0,loads=saves)


def extract_prot_seq_feat():
    featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7'])) 

if __name__ == '__main__':
    # extract_prot_seq_feat()
    execute()