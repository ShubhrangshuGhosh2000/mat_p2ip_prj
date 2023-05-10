import os
import sys
from pathlib import Path

import joblib
import pandas as pd

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from preproc.feat_engg_manual import feat_engg_manual_main


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

# prepare the manual features for the already saved DS_sequence list 
def prepare_manual_feat_for_DS_seq(root_path='./', spec_type = 'human'):
    print('\n########## spec_type: ' + str(spec_type))
    # fetch the already saved DS_sequence df
    print('\n ########## fetching the already saved DS_sequence df ########## ')
    DS_seq_df = pd.read_csv(os.path.join(root_path,'dataset/preproc_data_DS', 'DS_' + spec_type + '_seq.csv'))
    # extract the manual features for the DS_sequence list
    print('\n ########## extracting the manual features for the DS_sequence list ########## ')
    feature_type_lst =  ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
                        # , 'CHAOS' \
                        # , 'AAC20', 'AAC400', 'Grantham_Sequence_Order_30', 'Schneider_Sequence_Order_30' \
                        # # , 'DUMULTIGROUPCTD' \  # making the number of manual features too large
                        # , 'Grantham_Quasi_30', 'Schneider_Quasi_30', 'APSAAC30_2']
    # the dictionary to store the manual features
    DS_seq_manual_feat_dict = {}
    # iterate over the DS_seq_df and extract the manual features for each protein sequence 
    for index, row in DS_seq_df.iterrows():
        print('starting ' + str(index) + '-th protein out of ' + str(DS_seq_df.shape[0]))
        prot_id, prot_seq = row['prot_id'], row['seq']
        # extract the manual features
        seq_manual_feat_dict = feat_engg_manual_main.extract_prot_seq_manual_feat(root_path,
                                        prot_seq=prot_seq, feature_type_lst=feature_type_lst)
        # store them in DS_seq_manual_feat_dict 
        DS_seq_manual_feat_dict[prot_id] = {'seq': prot_seq, 'seq_manual_feat_dict': seq_manual_feat_dict}
    # save DS_seq_manual_feat_dict to a .pkl file
    print("\n Saving DS_seq_manual_feat_dict to a .pkl file...")
    filename = os.path.join(root_path, 'dataset/preproc_data_DS', 'DS_seq_manual_feat_dict_' + spec_type + '.pkl')
    joblib.dump(value=DS_seq_manual_feat_dict, filename=filename, compress=3)
    print("\n The DS_seq_manual_feat_dict is saved as: " + filename)


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    spec_type = 'human'  # human, ecoli, fly, mouse, worm, yeast 
    parse_DS_to_fasta(root_path, spec_type)

    prepare_manual_feat_for_DS_seq(root_path, spec_type)
