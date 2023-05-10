import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import umap
import numpy as np
from sklearn import preprocessing

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import preproc_util_DS


# parse the content of allSeqs.fasta and create a dataframe containing 'prot_id' and 'seq' columns
def parse_DS_to_fasta(root_path='./', spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    f = open(os.path.join(root_path, 'dataset/orig_data_DS/seqs', spec_type + '.fasta'))
    prot_lst, seq_lst, prot_len_lst = [], [], []
    idx = 0
    for line in f:
        if idx == 0:
            prot_lst.append(line.strip().strip('>'))
        elif idx == 1:
            seq_lst.append(line.strip())
        idx += 1
        idx = idx % 2
    f.close()

    # create dataframe
    DS_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    DS_seq_df['seq_len'] = DS_seq_df['seq'].str.len()

    # save DS_seq_df
    DS_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'), index=False)
    # return DS_seq_df
    return DS_seq_df

# add features generated by the protTrans model (tl model) to the already saved DS_sequence list 
def add_protTrans_feat_to_DS_seq(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    # fetch the already saved DS_sequence df
    print('\n ########## fetch the already saved DS_sequence df ########## ')
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'))
    # extract features using the protTrans model (tl model) for the DS_sequence list
    print('\n ########## extract features using the protTrans model (tl model) for the DS_sequence list ########## ')
    features_lst = preproc_util_DS.extract_feat_from_protTrans(DS_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name, spec_type)
    # use the extracted features alongwith DS_seq_df to create a dictionary to be saved as pkl file
    print('\n ########## use the extracted features alongwith DS_seq_df to create a dictionary to be saved as pkl file ########## ')
    DS_seq_feat_dict = {}
    for index, row in DS_seq_df.iterrows():
        DS_seq_feat_dict[row['prot_id']] = {'seq': row['seq'], 'seq_len': row['seq_len'], 'seq_feat': features_lst[index]}
    # save DS_seq_feat_dict to a .pkl file
    print("\n Saving DS_seq_feat_dict to a .pkl file...")
    filename = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_feat_dict_' + protTrans_model_name + '_' + spec_type + '.pkl')
    joblib.dump(value=DS_seq_feat_dict, filename=filename, compress=3)
    print("\n The DS_seq_feat_dict is saved as: " + filename)
    # print("\n######## cleaning all the intermediate stuffs - START ########")
    # # remove all the intermediate files in the 'temp_result' and 'temp_per_prot_emb_result' directories which
    # # were used in extract_feat_from_preloaded_protTrans() method
    # temp_result_dir = os.path.join('temp_result_' + spec_type) 
    # for temp_file in os.listdir(temp_result_dir):
    #     os.remove(os.path.join(temp_result_dir, temp_file))
    # temp_per_prot_emb_result_dir = os.path.join('temp_per_prot_emb_result_' + spec_type) 
    # for temp_file in os.listdir(temp_per_prot_emb_result_dir):
    #     os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))
    # print("######## cleaning all the intermediate stuffs - DONE ########")


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast 
    parse_DS_to_fasta(root_path, spec_type)

    # ## preproc_util.extract_feat_from_protTrans(["A E T C Z A O", "S K T Z P"], 
    # ## protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/', protTrans_model_name = 'prot_t5_xl_uniref50')
    add_protTrans_feat_to_DS_seq(root_path
                                    ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                    , protTrans_model_name = 'prot_t5_xl_uniref50'
                                    , spec_type = spec_type)
