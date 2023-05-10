import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
print(sys.path)

from utils import preproc_util

# prepare the dataset by the feature extraction by the tl-model after concatenation of the PPI sequence pair for the given dataset type
def data_prep_feat_post_seq_concat(root_path='./', dataset_type='Random50', protTrans_model_name = 'prot_t5_xl_uniref50'):
    print("\n ############ inside the data_prep_feat_post_seq_concat() method ############")
    input_path = os.path.join(root_path, 'dataset/orig_data/Human2021/BioGRID2021', dataset_type)
    output_path = os.path.join(root_path, 'dataset/preproc_data/human_2021', dataset_type, 'feat_post_seq_concat')
    print('input dataset path: ' + str(input_path))
    # filter the directory-list to include only the .tsv file names
    tsv_file_names_lst = glob.glob(os.path.join(input_path, '*.tsv'))
    print('tsv_file_names_lst: ' + str(tsv_file_names_lst))
    # load the pkl file containing the PPI sequences in a dictionary
    human_seq_feat_dict = joblib.load(os.path.join(root_path, 'dataset/preproc_data','human_seq_feat_dict_prot_t5_xl_uniref50.pkl'))
    # load the vocabulary and ProtTrans Model which will be used repeatedly later
    protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
    model, tokenizer = preproc_util.load_protTrans_tl_model(protTrans_model_path, protTrans_model_name)
    # iterate over the tsv files and perform the feature extraction by the tl-model after concatenation of the PPI sequence pair
    for indiv_tsv_file_nm in tsv_file_names_lst:
        indiv_tsv_file_nm = 'Train_0.tsv'  # ################## HARD CODED TO BE REMOVED ####################################
        print('\n indiv_tsv_file_nm: ' + str(indiv_tsv_file_nm))
        # derive the individual tsv file name without the extension
        indiv_tsv_file_nm_without_ext = indiv_tsv_file_nm.split('/')[-1].replace('.tsv', '')
        # check if the corresponding pkl file already exists in the respective output folder
        outp_file_nm = os.path.join(output_path, indiv_tsv_file_nm_without_ext + '.pkl')
        if(os.path.exists(outp_file_nm)):
            # the pkl file already exists
            print('corresponding pkl file (' + str(outp_file_nm) + ') already exists, hence skipping this tsv file...\n')
            continue
        indiv_tsv_df = pd.read_csv(os.path.join(input_path, indiv_tsv_file_nm), header=None, sep='\t')
        # iterate row-wise for each PPI
        concat_prot_pair_seq_lst = []
        for ind, row in indiv_tsv_df.iterrows():
            if(ind in [50000, 75000, 99999, 150000, 199999] or ind > 199999):
                print('processing row# ' +str(ind) + ' out of ' + str(indiv_tsv_df.shape[0]) + ' rows' )
            prot_1_id = row[0]  # 0-th column
            prot_2_id = row[1]  # 1-th column
            prot_1_seq = human_seq_feat_dict[prot_1_id]['seq']
            prot_2_seq = human_seq_feat_dict[prot_2_id]['seq']
            concat_prot_pair_seq = prot_1_seq + prot_2_seq
            # append to concat_prot_pair_seq_lst
            concat_prot_pair_seq_lst.append(concat_prot_pair_seq)
        # end of for loop
        # perform the feature extraction by the tl-model on the list of concatenated PPI sequence pairs
        # use preloaded ProtTrans model(s) for the feature extraction
        # each element of the prot_pair_feat_lst is a 1d array containing 1024 elements
        prot_pair_feat_lst,  prot_pair_feat_2d_lst = preproc_util.extract_feat_from_preloaded_protTrans(concat_prot_pair_seq_lst, model, tokenizer)

        # # convert the extracted features as list of 1d arrays
        # prot_pair_feat_lst_1d = [feat_2d_arr[0] for feat_2d_arr in prot_pair_feat_lst]  # list of 1d arrays

        # declare a list of 1d arrays to be saved as pkl file
        wrapper_feat_lst = []
        # again iterate row-wise for each PPI
        for ind, row in indiv_tsv_df.iterrows():
            wrapper_feat_lst.append(np.concatenate((row.to_numpy(), prot_pair_feat_lst[ind]), axis=0))
        # # create column names list
        # col_nm_lst = ['prot_1_id', 'prot_2_id', 'label']
        # for idx in range(3, wrapper_feat_lst[0].size):  # the first three column names are already considered
        #     col_nm_lst.append('feat_' + str(idx))
        # # create a dataframe from wrapper_feat_lst
        # wrapper_feat_df = pd.DataFrame(data=wrapper_feat_lst, columns=col_nm_lst)

        # save the wrapper_feat_lst as .pkl file
        joblib.dump(value=wrapper_feat_lst, filename=outp_file_nm, compress=3)
        print("the wrapper_feat_lst is saved as: " + outp_file_nm)
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
    # end of for loop: for indiv_tsv_file_nm in tsv_file_names_lst:

if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # dataset_type_lst = ['HeldOut20', 'HeldOut50', 'Random20', 'Random50']
    dataset_type_lst = ['Random50']

    # iteratively invoke the data preparation method for all the dataset types
    for dataset_type in dataset_type_lst:
        data_prep_feat_post_seq_concat(root_path, dataset_type, protTrans_model_name = 'prot_t5_xl_uniref50')

    