import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import feat_engg_manual_main


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

# prepare the manual features for the already saved AD_sequence list 
def prepare_manual_feat_for_AD_seq(root_path='./'):
    # fetch the already saved AD_sequence df
    print('\n ########## fetching the already saved AD_sequence df ########## ')
    AD_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_AD', 'AD_seq.csv'))
    # extract the manual features for the AD_sequence list
    print('\n ########## extracting the manual features for the AD_sequence list ########## ')
    feature_type_lst =  ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
    # the dictionary to store the manual features
    AD_seq_manual_feat_dict = {}
    # iterate over the AD_seq_df and extract the manual features for each protein sequence 
    for index, row in AD_seq_df.iterrows():
        print('starting ' + str(index) + '-th protein out of ' + str(AD_seq_df.shape[0]))
        prot_id, prot_seq = row['prot_id'], row['seq']
        # extract the manual features
        seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_1D_manual_feat(root_path,
                                        prot_seq=prot_seq, feature_type_lst=feature_type_lst)
        # store them in AD_seq_manual_feat_dict 
        AD_seq_manual_feat_dict[prot_id] = {'seq': prot_seq, 'seq_manual_feat_dict': seq_manual_feat_dict}
    # save AD_seq_manual_feat_dict to a .pkl file
    print("\n Saving AD_seq_manual_feat_dict to a .pkl file...")
    filename = os.path.join(root_path, 'dataset/preproc_data_AD', 'AD_seq_manual_feat_dict.pkl')
    joblib.dump(value=AD_seq_manual_feat_dict, filename=filename, compress=3)
    print("\n The AD_seq_manual_feat_dict is saved as: " + filename)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mtf_p2ip_prj')

    # # parse_AD_to_fasta(root_path)
    prepare_manual_feat_for_AD_seq(root_path)
