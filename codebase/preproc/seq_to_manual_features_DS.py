import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils import feat_engg_manual_main


def parse_DS_to_fasta(root_path='./', spec_type = 'human'):
    f = open(os.path.join(root_path, 'dataset/orig_data_DS/seqs', spec_type + '.fasta'))
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
    
    DS_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    DS_seq_df['seq_len'] = DS_seq_df['seq'].str.len()
    DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'), index=False)
    return DS_seq_df


def prepare_manual_feat_for_DS_seq(root_path='./', spec_type = 'human'):
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'))
    feature_type_lst =  ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    DS_seq_manual_feat_dict = {}
    
    for index, row in DS_seq_df.iterrows():
        prot_id, prot_seq = row['prot_id'], row['seq']
        seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path,
                                        prot_seq=prot_seq, feature_type_lst=feature_type_lst)
        DS_seq_manual_feat_dict[prot_id] = {'seq': prot_seq, 'seq_manual_feat_dict': seq_manual_feat_dict}
    
    filename = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_manual_feat_dict_' + spec_type + '.pkl')
    joblib.dump(value=DS_seq_manual_feat_dict, filename=filename, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    

    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast 
    parse_DS_to_fasta(root_path, spec_type)
    prepare_manual_feat_for_DS_seq(root_path, spec_type)
