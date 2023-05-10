import os

import joblib
import numpy as np
import pandas as pd


# prepare the dataset by the feature multiplication for the given dataset type using the PIPR architecture
def data_prep_feat_multiplied_pipr(root_path='./', dataset_type='Random50', tsv_file_nm='Train_0.tsv'):
    print("\n ############ inside the data_prep_feat_multiplied_pipr() method ############")
    input_path = os.path.join(root_path, 'dataset/orig_data/Human2021/BioGRID2021', dataset_type)
    print('input dataset path: ' + str(input_path))
    # load the pkl file containing the tl-model extracted 2d features post regularization for the proteins  in a dictionary
    human_seq_feat_2d_reg_dict = joblib.load(os.path.join(root_path, 'dataset/preproc_data','human_seq_feat_2d_reg_dict.pkl'))
    # iterate over the tsv file and perform the following:
    # create 3 arrays for each train or test tsv file, namely i) two_d_feat_arr_1, ii) two_d_feat_arr_2 and iii) label_arr_1d.
    # i. two_d_feat_arr_1 is an array of 2d arrays where each 2d array represents the pre-trained 2d embeddings of the 1st participating
    # protein of the PPI pairs.
    # ii. similarly two_d_feat_arr_2 is for the 2nd participating protein of the PPI pairs.
    # iii. label_arr_1d is an 1d array containing the respective labels
    print('\n tsv_file_nm: ' + str(tsv_file_nm))
    indiv_tsv_df = pd.read_csv(os.path.join(input_path, tsv_file_nm), header=None, sep='\t')
    # declare 3 lists corresponding to the 3 arrays to be created as mentioned above.
    two_d_feat_arr_1_lst, two_d_feat_arr_2_lst, label_arr_1d_lst = [], [], []
    # iterate row-wise for each PPI
    for ind, row in indiv_tsv_df.iterrows():
        if(ind in [50000, 75000, 99999, 150000, 199999] or ind > 199999):
            print('processing row# ' +str(ind) + ' out of ' + str(indiv_tsv_df.shape[0]) + ' rows' )
        prot_1_id, prot_2_id, label = row[0], row[1], row[2]
        # retrieve the correponding two_d_feat_arr from human_seq_feat_2d_reg_dict
        prot_1_two_d_feat_arr = human_seq_feat_2d_reg_dict[prot_1_id]
        prot_2_two_d_feat_arr = human_seq_feat_2d_reg_dict[prot_2_id]
        # append prot_1_two_d_feat_arr, prot_2_two_d_feat_arr and label to the corresponding lists
        two_d_feat_arr_1_lst.append(prot_1_two_d_feat_arr)
        two_d_feat_arr_2_lst.append(prot_2_two_d_feat_arr)
        label_arr_1d_lst.append(label)

    # convert two_d_feat_arr_1_lst, two_d_feat_arr_2_lst and label_arr_1d_lst into numpy arrays and return them
    print("converting two_d_feat_arr_1_lst, two_d_feat_arr_2_lst and label_arr_1d_lst into numpy arrays and returning them")
    two_d_feat_arr_1 = np.array(two_d_feat_arr_1_lst)
    two_d_feat_arr_2 = np.array(two_d_feat_arr_2_lst)
    label_arr_1d = np.array(label_arr_1d_lst)

    return (two_d_feat_arr_1, two_d_feat_arr_2, label_arr_1d)

if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    # dataset_type_lst = ['HeldOut20', 'HeldOut50', 'Random20', 'Random50']
    data_prep_feat_multiplied_pipr(root_path, dataset_type='Random50', tsv_file_nm='Train_0.tsv')

    