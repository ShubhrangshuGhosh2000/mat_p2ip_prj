import glob
import os

import joblib
import numpy as np
import pandas as pd


# prepare the dataset by the manual feature concatenation for the given dataset type
def prep_man_feat_side_by_side(root_path='./', dataset_type='Random50'):
    print("\n ############ inside the prep_man_feat_side_by_side() method ############")
    input_path = os.path.join(root_path, 'dataset/orig_data/Human2021/BioGRID2021', dataset_type)
    output_path = os.path.join(root_path, 'dataset/preproc_data/human_2021_manual', dataset_type, 'feat_side_by_side')
    print('input dataset path: ' + str(input_path))
    # filter the directory-list to include only the .tsv file names
    tsv_file_names_lst = glob.glob(os.path.join(input_path, '*.tsv'))
    print('tsv_file_names_lst: ' + str(tsv_file_names_lst))
    # load the pkl file containing the extracted manual features for the proteins in a dictionary
    human_seq_manual_feat_dict = joblib.load(os.path.join(root_path, 'dataset/preproc_data','human_seq_manual_feat_dict.pkl'))
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
        wrapper_man_feat_lst = []
        # iterate row-wise for each PPI
        for ind, row in indiv_tsv_df.iterrows():
            if(ind in [50000, 75000, 99999, 150000, 199999] or ind > 199999):
                print('processing row# ' +str(ind) + ' out of ' + str(indiv_tsv_df.shape[0]) + ' rows' )
            prot_1_id = row[0]
            prot_2_id = row[1]
            # processing for the protein-1
            seq_manual_feat_dict = human_seq_manual_feat_dict[prot_1_id]['seq_manual_feat_dict']
            # concatenate all the different types of manual features for protein-1
            prot_1_man_feat_lst = seq_manual_feat_dict['AC30'] + seq_manual_feat_dict['PSAAC15'] + seq_manual_feat_dict['ConjointTriad'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] + seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] \
                                + seq_manual_feat_dict['CHAOS'] \
                                + seq_manual_feat_dict['AAC20'] + seq_manual_feat_dict['AAC400'] \
                                + seq_manual_feat_dict['Grantham_Sequence_Order_30'] + seq_manual_feat_dict['Schneider_Sequence_Order_30'] \
                                + seq_manual_feat_dict['Grantham_Quasi_30'] + seq_manual_feat_dict['Schneider_Quasi_30'] + seq_manual_feat_dict['APSAAC30_2']
                                # + seq_manual_feat_dict['DuMultiCTD_C'] + seq_manual_feat_dict['DuMultiCTD_T'] + seq_manual_feat_dict['DuMultiCTD_D']

            prot_1_man_feat_arr = np.array(prot_1_man_feat_lst)
            # processing for the protein-2
            seq_manual_feat_dict = human_seq_manual_feat_dict[prot_2_id]['seq_manual_feat_dict']
            # concatenate all the different types of manual features for protein-2
            prot_2_man_feat_lst = seq_manual_feat_dict['AC30'] + seq_manual_feat_dict['PSAAC15'] + seq_manual_feat_dict['ConjointTriad'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] + seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] \
                                + seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] \
                                + seq_manual_feat_dict['CHAOS'] \
                                + seq_manual_feat_dict['AAC20'] + seq_manual_feat_dict['AAC400'] \
                                + seq_manual_feat_dict['Grantham_Sequence_Order_30'] + seq_manual_feat_dict['Schneider_Sequence_Order_30'] \
                                + seq_manual_feat_dict['Grantham_Quasi_30'] + seq_manual_feat_dict['Schneider_Quasi_30'] + seq_manual_feat_dict['APSAAC30_2']
                                # + seq_manual_feat_dict['DuMultiCTD_C'] + seq_manual_feat_dict['DuMultiCTD_T'] + seq_manual_feat_dict['DuMultiCTD_D']

            prot_2_man_feat_arr = np.array(prot_2_man_feat_lst)
            # concatenate the feature list for both the proteins side-by-side
            wrapper_man_feat_lst.append(np.concatenate((row.to_numpy(), prot_1_man_feat_arr, prot_2_man_feat_arr), axis=0))
        # # create column names list
        # col_nm_lst = ['prot_1_id', 'prot_2_id', 'label']
        # for idx in range(3, len(wrapper_man_feat_lst[0])):  # the first three column names are already considered
        #     col_nm_lst.append('feat_' + str(idx))
        # # create a dataframe from wrapper_man_feat_lst
        # wrapper_man_feat_df = pd.DataFrame(data=wrapper_man_feat_lst, columns=col_nm_lst)
         
        # save the wrapper_man_feat_lst as .pkl file
        joblib.dump(value=wrapper_man_feat_lst, filename=outp_file_nm, compress=0)
        print("the wrapper_man_feat_lst is saved as: " + outp_file_nm)


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # dataset_type_lst = ['Random50', 'HeldOut50', 'Random20', 'HeldOut20']
    dataset_type_lst = ['HeldOut20']

    # iteratively invoke the data preparation method for all the dataset types
    for dataset_type in dataset_type_lst:
        prep_man_feat_side_by_side(root_path, dataset_type)

    