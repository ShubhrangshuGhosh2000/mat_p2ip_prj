import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import preproc_tl_util


# parse the content of allSeqs.fasta and create a dataframe containing 'prot_id' and 'seq' columns
def parse_AD_to_fasta(root_path='./'):
    f = open(os.path.join(root_path, 'dataset/orig_data_AD/Li_AD', 'allSeqs.fasta'))
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
    AD_seq_df = pd.DataFrame(data = {'prot_id': prot_lst, 'seq': seq_lst})
    AD_seq_df['seq_len'] = AD_seq_df['seq'].str.len()

    # save AD_seq_df
    AD_seq_df.to_csv(os.path.join(root_path, 'dataset/preproc_data_AD', 'AD_seq.csv'), index=False)
    # return AD_seq_df
    return AD_seq_df


# add features generated by the protTrans model (tl model) to the already saved AD_sequence list 
def prepare_tl_feat_for_AD_seq(root_path='./', protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'):
    # fetch the already saved AD_sequence df
    print('\n ########## fetch the already saved AD_sequence df ########## ')
    AD_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_AD', 'AD_seq.csv'))
    # extract features using the protTrans model (tl model) for the AD_sequence list
    print('\n ########## extract features using the protTrans model (tl model) for the AD_sequence list ########## ')
    features_lst, features_2d_lst = preproc_tl_util.extract_feat_from_protTrans(AD_seq_df['seq'].tolist(), protTrans_model_path, protTrans_model_name)
    # use the extracted features alongwith AD_seq_df to create a dictionary to be saved as pkl file
    print('\n ########## use the extracted features alongwith AD_seq_df to create a dictionary to be saved as pkl file ########## ')
    AD_seq_feat_dict = {}
    for index, row in AD_seq_df.iterrows():
        AD_seq_feat_dict[row['prot_id']] = {'seq': row['seq'], 'seq_len': row['seq_len'], 'seq_feat': features_lst[index], 'seq_2d_feat': features_2d_lst[index]}
    # save AD_seq_feat_dict to a .pkl file
    print("\n Saving AD_seq_feat_dict to a .pkl file...")
    filename = os.path.join(root_path, 'dataset/preproc_data_AD', 'AD_seq_feat_dict_' + protTrans_model_name + '.pkl')
    joblib.dump(value=AD_seq_feat_dict, filename=filename, compress=3)
    print("\n The AD_seq_feat_dict is saved as: " + filename)
    print("\n######## cleaning all the intermediate stuffs - START ########")
    # remove all the intermediate files in the 'temp_result' and 'temp_per_prot_emb_result' directories which
    # were used in extract_feat_from_preloaded_protTrans() method
    temp_result_dir = os.path.join('temp_result') 
    for temp_file in os.listdir(temp_result_dir):
        os.remove(os.path.join(temp_result_dir, temp_file))
    temp_per_prot_emb_result_dir = os.path.join('temp_per_prot_emb_result') 
    for temp_file in os.listdir(temp_per_prot_emb_result_dir):
        os.remove(os.path.join(temp_per_prot_emb_result_dir, temp_file))
    print("######## cleaning all the intermediate stuffs - DONE ########")


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    # parse_AD_to_fasta(root_path)
    prepare_tl_feat_for_AD_seq(root_path
                                    ,protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
                                    , protTrans_model_name = 'prot_t5_xl_uniref50')