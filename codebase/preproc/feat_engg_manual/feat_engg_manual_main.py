# this is the main module for manual feature extraction/engineering
import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from preproc.feat_engg_manual.AutoCovariance import AutoCovariance
from preproc.feat_engg_manual.ConjointTriad import ConjointTriad
from preproc.feat_engg_manual.LDCTD import LDCTD
from preproc.feat_engg_manual.PSEAAC import PSEAAC

def extract_prot_seq_manual_feat(root_path='./', prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE', feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD'], deviceType='cpu'):
    print('#### inside the extract_prot_seq_manual_feat() method - Start')
    fastas = [('999', prot_seq)]
    featureSets = set(feature_type_lst)
    # the dictionary to be returned
    seq_manual_feat_dict = {}

    if 'AC30' in featureSets:
        print("Calculating 'AC30' feature - Start")
        ac = AutoCovariance(fastas, lag=30, deviceType=deviceType)    
        seq_manual_feat_dict['AC30'] = ac[1][1:]
        print("Calculating 'AC30' feature - End")

    if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
        print("Calculating 'PSAAC15' feature - Start")
        paac = PSEAAC(fastas,lag=15)
        seq_manual_feat_dict['PSAAC15'] = paac[1][1:]
        print("Calculating 'PSAAC15' feature - End")

    if 'ConjointTriad' in featureSets or 'CT' in featureSets:
        print("Calculating 'ConjointTriad' feature - Start")
        ct = ConjointTriad(fastas,deviceType=deviceType)
        seq_manual_feat_dict['ConjointTriad'] = ct[1][1:]
        print("Calculating 'ConjointTriad' feature - End")

    if 'LD10_CTD' in featureSets:
        print("Calculating 'LD10_CTD' feature - Start")
        (comp, tran, dist) = LDCTD(fastas)
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] = comp[1][1:]
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] = tran[1][1:]
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] = dist[1][1:]
        print("Calculating 'LD10_CTD' feature - End")

    print('#### inside the extract_prot_seq_manual_feat() method - End')
    return seq_manual_feat_dict


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE'
    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    seq_manual_feat_dict = extract_prot_seq_manual_feat(root_path, prot_seq = prot_seq, feature_type_lst = feature_type_lst, deviceType='cpu')

