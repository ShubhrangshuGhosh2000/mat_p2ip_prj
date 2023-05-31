import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))
from utils import feat_engg_manual_main


def parse_human_to_fasta(root_path='./'):
    f = open(os.path.join(root_path, 'dataset/orig_data/Human2021', 'allSeqs.fasta'))
    prot_lst, seq_lst = [], []
    idx = 0
    for line in f:
        if idx == 0:
            prot_lst.append(line.strip().strip('>'))
        elif idx == 1:
            seq_lst.append(line.strip())
        idx += 1
        idx = idx % 2
    f.close()
    
    human_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    human_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data', 'human_seq.csv'), index=False)
    return human_seq_df


def prepare_manual_feat_for_human_seq(root_path='./'):
    human_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data', 'human_seq.csv'))
    feature_type_lst =  ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    human_seq_manual_feat_dict = {}
    
    for index, row in human_seq_df.iterrows():
        prot_id, prot_seq = row['prot_id'], row['seq']
        seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path,
                                        prot_seq=prot_seq, feature_type_lst=feature_type_lst)
        
        human_seq_manual_feat_dict[prot_id] = {'seq': prot_seq, 'seq_manual_feat_dict': seq_manual_feat_dict}
    
    filename = os.path.join(root_path, 'dataset/preproc_data', 'human_seq_manual_feat_dict.pkl')
    joblib.dump(value=human_seq_manual_feat_dict, filename=filename, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

    # ### First remove all the references of the protein id = 7273 from everywhere (allSeqs.fasta, training sets,
    #  test sets, etc.). You can call remove_7273() method from preproc_util.py file.
    # ## parse_human_to_fasta(root_path)

    prepare_manual_feat_for_human_seq(root_path)
