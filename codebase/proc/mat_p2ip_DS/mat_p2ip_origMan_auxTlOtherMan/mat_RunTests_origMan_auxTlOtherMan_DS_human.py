import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_DS_train import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_DS


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')


def execute(spec_type = 'human'): 
    resultsFolderName = os.path.join(root_path, 'dataset/proc_data_DS/mat_res_origMan_auxTlOtherMan_' + spec_type + '/')
    
    PPIPUtils.makeDir(resultsFolderName)

    # MUST SET ENV VARIABLE 'CUDA_VISIBLE_DEVICES' in the runtime environment where multigpu training will take place:
    # export CUDA_VISIBLE_DEVICES=0,1
    # echo $CUDA_VISIBLE_DEVICES
    hyp = {'fullGPU':True, 'deviceType':'cuda'} 

    hyp['maxProteinLength'] = 800  
    
    hyp['hiddenSize'] = 70  
    hyp['numLayers'] = 4 
    hyp['n_heads'] = 1 
    hyp['layer_1_size'] = 1024  

    hyp['batchSize'] = 256  
    hyp['numEpochs'] = 150  

    trainSets, testSets, saves, pfs, featureFolder = loadDscriptData_human_full(resultsFolderName)
    mat_RunTrainTest_origMan_auxTlOtherMan_DS.runTrainOnly_DS(MatP2ipNetworkModule, trainSets, featureFolder, hyp, saves, spec_type)


def extract_prot_seq_feat(spec_type = 'human'):
    featureDir = os.path.join(root_path, 'dataset/preproc_data_DS/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+spec_type+'/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7']), spec_type = spec_type) 


if __name__ == '__main__':
    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast
    # extract_prot_seq_feat(spec_type)
    execute(spec_type)