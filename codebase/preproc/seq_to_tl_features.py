import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils import preproc_tl_util


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


def prepare_tl_feat_for_human_seq(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'):
    human_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data', 'human_seq.csv'))
    features_lst, features_2d_lst = preproc_tl_util.extract_feat_from_protTrans(human_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name)
    human_seq_feat_dict = {}
    for index, row in human_seq_df.iterrows():
        human_seq_feat_dict[row['prot_id']] = {'seq': row['seq'], 'seq_feat': features_lst[index], 'seq_2d_feat': features_2d_lst[index]}
    
    filename = os.path.join(root_path, 'dataset/preproc_data', 'human_seq_feat_dict_' + protTrans_model_name + '.pkl')
    joblib.dump(value=human_seq_feat_dict, filename=filename, compress=3)
    
    temp_result_dir = os.path.join('temp_result') 
    for temp_file in os.listdir(temp_result_dir):
        os.remove(os.path.join(temp_result_dir, temp_file))
    temp_per_prot_emb_result_dir = os.path.join('temp_per_prot_emb_result') 
    for temp_file in os.listdir(temp_per_prot_emb_result_dir):
        os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

    # ### First remove all the references of the protein id = 7273 from everywhere (allSeqs.fasta, training sets,
    #  test sets, etc.). You can call remove_7273() method from preproc_util.py file.
    # ## parse_human_to_fasta(root_path)

    prepare_tl_feat_for_human_seq(root_path
                                    ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                    , protTrans_model_name = 'prot_t5_xl_uniref50')
