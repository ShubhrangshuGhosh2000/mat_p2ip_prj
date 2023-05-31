import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils import preproc_tl_util_DS


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


def prepare_tl_feat_for_DS_seq(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', spec_type = 'human'):
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'))
    features_lst = preproc_tl_util_DS.extract_feat_from_protTrans(DS_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name, spec_type)
    
    DS_seq_feat_dict = {}
    for index, row in DS_seq_df.iterrows():
        DS_seq_feat_dict[row['prot_id']] = {'seq': row['seq'], 'seq_len': row['seq_len'], 'seq_feat': features_lst[index]}
    filename = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_feat_dict_' + protTrans_model_name + '_' + spec_type + '.pkl')
    joblib.dump(value=DS_seq_feat_dict, filename=filename, compress=3)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast 
    parse_DS_to_fasta(root_path, spec_type)

    prepare_tl_feat_for_DS_seq(root_path
                                    ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                    , protTrans_model_name = 'prot_t5_xl_uniref50'
                                    , spec_type = spec_type)
