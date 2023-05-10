import os

import joblib
import numpy as np
from sklearn import preprocessing
import umap


# reduce the full vesion of the manual feature dataset for the given
# dataset type (e.g. 'Random50') using scaling followed by the dimensionality reduction
def reduc_man_feat_sbs_random50(root_path='./'):
    print("\n ############ inside the reduc_man_feat_sbs_random50() method -Start ############")
    dataset_type='Random50'
    reduc_dim = 10
    inp_outp_path = os.path.join(root_path, 'dataset/preproc_data/human_2021_manual', dataset_type, 'feat_side_by_side')

    # dataset_type specific processing
    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(0, 1):
        train_pkl_name = 'Train_' + str(ind) + '.pkl'
        test_pkl_name = train_pkl_name.replace('Train', 'Test')
        # first check whether the test file is already dimensionally reduced
        test_pkl_name_without_ext = test_pkl_name.replace('.pkl', '') 
        reduced_test_pkl_file_path = os.path.join(inp_outp_path, test_pkl_name_without_ext + '_reduced_' + str(reduc_dim) + '.pkl')
        if os.path.exists(reduced_test_pkl_file_path):
            # as the test file is already dimensionally reduced, skipping this train-test combination
            print('\n##### As the test file "' + str(test_pkl_name) + '" is already dimensionally reduced, skipping this train-test combination...')
            continue
        # load train the pkl file
        print('\n ##### processing the training data with train_pkl_name: ' + str(train_pkl_name) + '\n')
        train_lst = joblib.load(os.path.join(inp_outp_path, train_pkl_name))
        # train_lst is a lst of 1d arrays; now convert it into a 2d array
        train_arr_2d = np.vstack(train_lst)
        # next perform column filtering and column rearranging so that the feature columns come first and then
        # the target column (in train_arr_2d, the target column is in the 2th column index and features are started
        # from 3th column index onwards)
        train_arr = train_arr_2d[:, list(range(3, train_arr_2d.shape[1])) + [2]]
        X_train_arr = train_arr[:, range(0, train_arr.shape[1] -1)]  # excluding the target column
        y_train_arr = train_arr[:, -1]  # the last column i.e. target column
        # z-normalize X_train_arr feature(column) wise
        print('z-normalizing X_train_arr manual feature(column) wise')
        scaler = preprocessing.StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_arr)
        # perform the dimensionality reduction on the training data
        print('performing the dimensionality reduction on the training data')
        print('reduc_dim = ' + str(reduc_dim))
        # initiate the umap
        dim_reducer = umap.UMAP(n_components=reduc_dim, low_memory=False, n_epochs=400, densmap=False
                                , n_neighbors=30, random_state=456, verbose=True)
        # apply fit on the X_train_scaled and y_train_arr
        dim_reducer.fit(X_train_scaled, y_train_arr)
        # apply transform on the X_train_scaled
        X_train_scaled_reduced = dim_reducer.transform(X_train_scaled)

        print('\n ##### processing the test data with test_pkl_name: ' + str(test_pkl_name) + '\n')
        # Also scale and dimensionally reduce the test data using the same scaler and reducer used to scale and
        # reduce the taining data and save them so that during the testing they can be used directly
        test_lst = joblib.load(os.path.join(inp_outp_path, test_pkl_name))
        # test_lst is a lst of 1d arrays; now convert it into a 2d array
        test_arr_2d = np.vstack(test_lst)
        # next perform column filtering and column rearranging so that the feature columns come first and then
        # the target column (in train_arr_2d, the target column is in the 2th column index and features are started
        # from 3th column index onwards)
        test_arr = test_arr_2d[:, list(range(3, test_arr_2d.shape[1])) + [2]]
        X_test_arr = test_arr[:, range(0, test_arr.shape[1] -1)]  # excluding the target column
        y_test_arr = test_arr[:, -1]  # the last column i.e. target column
        # z-normalize X_test_arr feature(column) wise
        X_test_scaled = scaler.transform(X_test_arr)
        # apply transform on the X_test_scaled
        X_test_scaled_reduced = dim_reducer.transform(X_test_scaled)
        # save the dimensionally reduced result
        print('\n saving the dimensionally reduced result')
        print('saving reduced_train_pkl')
        train_pkl_name_without_ext = train_pkl_name.replace('.pkl', '') 
        reduced_train_pkl_file_path = os.path.join(inp_outp_path, train_pkl_name_without_ext + '_reduced_' + str(reduc_dim) + '.pkl')
        joblib.dump(value={'X_train_scaled_reduced': X_train_scaled_reduced, 'y_train_arr': y_train_arr}
                    , filename=reduced_train_pkl_file_path, compress=3)
        print('saving reduced_test_pkl')
        joblib.dump(value={'X_test_scaled_reduced': X_test_scaled_reduced, 'y_test_arr': y_test_arr}
                    , filename=reduced_test_pkl_file_path, compress=3)
    # end of for loop
    print("\n ############ inside the reduc_man_feat_sbs_random50() method -End ############")


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    reduc_man_feat_sbs_random50(root_path)

    