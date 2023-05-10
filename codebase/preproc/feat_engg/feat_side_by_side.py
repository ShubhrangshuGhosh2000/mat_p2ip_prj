import glob
import os

import joblib
import numpy as np
import dask.dataframe as pd


# prepare the dataset by the feature concatenation for the given dataset type
def data_prep_feat_side_by_side(root_path='./', dataset_type='Random50'):
    print("\n ############ inside the data_prep_feat_side_by_side() method ############")
    input_path = os.path.join(root_path, 'dataset/orig_data/Human2021/BioGRID2021', dataset_type)
    output_path = os.path.join(root_path, 'dataset/preproc_data/human_2021', dataset_type, 'feat_side_by_side')
    print('input dataset path: ' + str(input_path))
    # filter the directory-list to include only the .tsv file names
    tsv_file_names_lst = glob.glob(os.path.join(input_path, '*.tsv'))
    print('tsv_file_names_lst: ' + str(tsv_file_names_lst))
    # load the pkl file containing the tl-model extracted features for the proteins in a dictionary
    human_seq_feat_dict = joblib.load(os.path.join(root_path, 'dataset/preproc_data','human_seq_feat_dict_prot_t5_xl_uniref50.pkl'))
    # iterate over the tsv files and perform the feature concatenation
    for indiv_tsv_file_nm in tsv_file_names_lst:
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
        # declare a list of 1d arrays to be saved as pkl file
        wrapper_feat_lst = []
        # iterate row-wise for each PPI
        for ind, row in indiv_tsv_df.iterrows():
            if(ind in [50000, 75000, 99999, 150000, 199999] or ind > 199999):
                print('processing row# ' +str(ind) + ' out of ' + str(indiv_tsv_df.shape[0]) + ' rows' )
            prot_1_id = row[0]
            prot_2_id = row[1]
            prot_1_feat_arr = human_seq_feat_dict[prot_1_id]['seq_feat']
            prot_2_feat_arr = human_seq_feat_dict[prot_2_id]['seq_feat']
            # concatenate the feature list
            wrapper_feat_lst.append(np.concatenate((row.to_numpy(), prot_1_feat_arr, prot_2_feat_arr), axis=0))
        # # create column names list
        # col_nm_lst = ['prot_1_id', 'prot_2_id', 'label']
        # for idx in range(3, len(wrapper_feat_lst[0])):  # the first three column names are already considered
        #     col_nm_lst.append('feat_' + str(idx))
        # # create a dataframe from wrapper_feat_lst
        # wrapper_feat_df = pd.DataFrame(data=wrapper_feat_lst, columns=col_nm_lst)
         
        # save the wrapper_feat_lst as .pkl file
        joblib.dump(value=wrapper_feat_lst,
                    filename=outp_file_nm,
                    compress=3)
        print("the wrapper_feat_lst is saved as: " + outp_file_nm)



if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    dataset_type_lst = ['HeldOut20', 'HeldOut50', 'Random20', 'Random50']

    # iteratively invoke the data preparation method for all the dataset types
    for dataset_type in dataset_type_lst:
        data_prep_feat_side_by_side(root_path, dataset_type)

    