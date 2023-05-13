import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from utils import PPIPUtils
from proc.mtf_p2ip.mtf_p2ip_origMan_auxTlOtherMan.mtf_MtfP2ipNetwork_origMan_auxTlOtherMan_rand20 import MtfP2ipNetworkModule
from utils.ProjectDataLoader import *
from utils.feat_engg_manual_main import extract_prot_seq_2D_manual_feat
from proc.mtf_p2ip.mtf_p2ip_origMan_auxTlOtherMan import mtf_RunTrainTest_origMan_auxTlOtherMan_rand20


root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mtf_p2ip_prj')

resultsFolderName = os.path.join(root_path, 'dataset/proc_data/mtf_res/mtf_res_origMan_auxTlOtherMan/')

def execute():
    #create results folders if they do not exist
    PPIPUtils.makeDir(resultsFolderName)
    hyp = {'fullGPU':True,'deviceType':'cuda'} 

    hyp['hiddenSize'] = 70  # default: 20
    hyp['numLayers'] = 4  # default: 6
    hyp['n_heads'] = 1  # default: 2
    hyp['layer_1_size'] = 1024  # default: 1024  # for the linear layers
    print('hyp: ' + str(hyp))

    trainSets, testSets, saves, pfs, folderName = loadHumanRandom20(resultsFolderName)
    outResultsNameLst = []
    outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_rand20_1.txt'))
    outResultsNameLst.append(os.path.join(resultsFolderName, 'mtf_res_origMan_auxTlOtherMan_rand20_2.txt'))
    # mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.runTestLst(MtfP2ipNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True, modelsLst=None,saveModels=convertToFolder(saves),predictionsFLst = pfs,startIdx=0)
    mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.runTestLst(MtfP2ipNetworkModule, outResultsNameLst,trainSets,testSets,folderName,hyp,resultsAppend=True, modelsLst=None,saveModels=convertToFolder(saves),predictionsFLst = pfs,startIdx=0)
    mtf_RunTrainTest_origMan_auxTlOtherMan_rand20.calcOverallScore_Pos20(outResultsNameLst)


def extract_prot_seq_feat():
    featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
    extract_prot_seq_2D_manual_feat(featureDir+'PPI_Datasets/Human2021/',set(['PSSM', 'LabelEncoding', 'Blosum62', 'SkipGramAA7'])) 

if __name__ == '__main__':
    # extract_prot_seq_feat()
    execute()