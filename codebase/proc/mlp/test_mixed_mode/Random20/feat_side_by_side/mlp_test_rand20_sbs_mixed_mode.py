import os
import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import torch
# import umap
# from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torch.nn import functional as F
import glob
from sklearn import metrics

path_root = Path(__file__).parents[5]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import test_util
from proc.mlp.train_mixed_mode.Random20.feat_side_by_side.mlp_train_rand20_sbs_fixed_lr_mixed_mode import MlpModelRand20Sbs_fixedLrTrain_MixedMode

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data_mixed_mode/Random20/feat_side_by_side'
TEST_RES_DIR = 'dataset/proc_data/mlp_data/test_data_mixed_mode/Random20/feat_side_by_side'

def load_final_ckpt_model(root_path='./', train_pkl_name = 'train.pkl'):
    print('#### inside the load_final_ckpt_model() method - Start')
    # create the final checkpoint path
    final_ckpt_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'fixed_lr_train_results')
    train_pkl_name_without_ext = train_pkl_name.replace('.pkl', '') 
    final_ckpt_path = os.path.join(final_ckpt_dir, "MlpModelRand20Sbs_fixedLrTrain_MixedMode_" + str(train_pkl_name_without_ext) + '*.ckpt')
    final_ckpt_file_name = glob.glob(final_ckpt_path, recursive=False)[0]
    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = MlpModelRand20Sbs_fixedLrTrain_MixedMode.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = MlpModelRand20Sbs_fixedLrTrain_MixedMode()
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)

    # restore back important config object from the saved pkl file
    imp_config_dict_pkl_path = os.path.join(root_path, CHECKPOINTS_DIR, 'fixed_lr_train_results', train_pkl_name.replace('.pkl', '') + '_imp_config_dict.pkl') 
    imp_config_dict = joblib.load(imp_config_dict_pkl_path)
    model.scaler = imp_config_dict['scaler']
    print('#### inside the load_final_ckpt_model() method - End')
    return model


def prepare_test_data(root_path='./', train_pkl_name = 'train.pkl', test_pkl_name = 'test.pkl', model = None):
    print('#### inside the prepare_test_data() method - Start')
    # extract the tl-based features for the test data
    print('\nExtracting the tl-based features for the test data...')
    tl_test_data_path = os.path.join(root_path, 'dataset/preproc_data/human_2021/Random20/feat_side_by_side', test_pkl_name)
    print('tl_test_data_path: ' + str(tl_test_data_path))
    # load the test pkl file
    test_lst_tl = joblib.load(tl_test_data_path)
    # test_lst_tl is a lst of 1d arrays; now convert it into a 2d array
    test_arr_2d_tl = np.vstack(test_lst_tl)
    # next perform column filtering and column rearranging so that the feature columns come first and then
    # the target column (in test_arr_2d_tl, the target column is in the 2th column index and features are started
    # from 3th column index onwards)
    test_arr_tl = test_arr_2d_tl[:, list(range(3, test_arr_2d_tl.shape[1])) + [2]]
    X_test_arr_tl = test_arr_tl[:, range(0, test_arr_tl.shape[1] -1)]  # excluding the target column
    y_test_arr_tl = test_arr_tl[:, -1]  # the last column i.e. target column
    print('X_test_arr_tl.shape: ' + str(X_test_arr_tl.shape))  # (, 2048)

    # extract the manual features for the test data
    print('\nExtracting the manual features for the test data...')
    manual_test_data_path = os.path.join(root_path, 'dataset/preproc_data/human_2021_manual/Random20/feat_side_by_side', test_pkl_name)
    print('manual_test_data_path: ' + str(manual_test_data_path))
    # load the test pkl file
    test_lst_manual = joblib.load(manual_test_data_path)
    # test_lst_manual is a lst of 1d arrays; now convert it into a 2d array
    test_arr_2d_manual = np.vstack(test_lst_manual)
    # next perform column filtering and column rearranging so that the feature columns come first and then
    # the target column (in test_arr_2d, the target column is in the 2th column index and features are started
    # from 3th column index onwards)
    test_arr_manual = test_arr_2d_manual[:, list(range(3, test_arr_2d_manual.shape[1])) + [2]]
    X_test_arr_manual = test_arr_manual[:, range(0, test_arr_manual.shape[1] -1)]  # excluding the target column
    y_test_arr_manual = test_arr_manual[:, -1]  # the last column i.e. target column
    print('X_test_arr_manual.shape: ' + str(X_test_arr_manual.shape))  # (, 2436)

    # concatenate the respective tl-based features and manual features
    print('\nConcatenating the respective tl-based features and manual features')
    X_test_concatenated = np.concatenate((X_test_arr_tl, X_test_arr_manual), axis=1)  # column-wise concatenation
    # z-normalize X_test_concatenated feature(column) wise using the scaler saved during the training data preparation
    X_test_scaled_reduced = model.scaler.transform(X_test_concatenated)
    y_test_arr = y_test_arr_tl
    print('X_test_scaled_reduced.shape: ' + str(X_test_scaled_reduced.shape))  # (, 4484)
    X_test, y_test = X_test_scaled_reduced, y_test_arr

    # tranform the 2d numpy arrays to the torch tensors
    print('transforming the 2d numpy arrays to the torch tensors')
    X_test = X_test.astype(np.float32)
    X_test = torch.from_numpy(X_test)
    # transform the 1d numpy array of floats to the array of int as the target label is integer
    # no need for np.round() here as the floats are either 0.0 or 1.0
    y_test = y_test.astype(int)
    print('#### inside the prepare_test_data() method - End')
    return (test_arr_2d_tl, X_test, y_test)


def test_model(root_path='./', train_pkl_name = 'train.pkl', test_pkl_name = 'test.pkl'):
    print('\n #### inside the test_model() method - Start\n')
    print('# #############################\n')
    print('train_pkl_name: ' + train_pkl_name + '\ntest_pkl_name: ' + test_pkl_name)
    print('\n# #############################')

    # create the test_result_dir if it does not already exist
    test_result_dir = os.path.join(root_path, TEST_RES_DIR)
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, train_pkl_name)
    # prepare the test data
    print('\n preparing the test data')
    test_arr_2d, X_test, y_test = prepare_test_data(root_path, train_pkl_name, test_pkl_name, model)
    # perform the prediction
    print('\n performing the prediction')
    model.eval()
    logits = None
    with torch.no_grad():
        logits = model(X_test)
    # find the predicted class (0 or 1)
    __, pred = torch.max(logits.data, 1)
    # find the prediction probability
    # logits contains the log of probability(i.e. softmax) in a 2d format where data-points are arranged
    # in rows and columns contain class-0 logit and class-1 logit. Its dimension is n x 2 where n is the
    # number of data points and there are 2 columns containing logits values for class-0 and class-1 respectively.
    prob_2d_arr = F.softmax(logits, dim=1)

    # create a df to store the prediction result
    print('creating a df to store the prediction result')
    actual_res = y_test
    pred_prob_0_arr = prob_2d_arr[:, 0]
    pred_prob_1_arr = prob_2d_arr[:, 1]
    pred_result_df = pd.DataFrame({'prot_1': test_arr_2d[:,0], 'prot_2': test_arr_2d[:,1]
                                    , 'actual_res': actual_res
                                    , 'pred_res': pred.numpy()
                                    , 'pred_prob_0': pred_prob_0_arr
                                    , 'pred_prob_1': pred_prob_1_arr
                                    })
    # save the pred_result_df
    test_pkl_name_without_ext = test_pkl_name.replace('.pkl', '') 
    file_nm_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random20_' + test_pkl_name_without_ext + '_pred_res.csv')
    pred_result_df.to_csv(file_nm_with_loc, index=False)

    # generate the performance metrics for the prediction - sk-learn way - Start
    print('generate the performance metrics for the prediction - sk-learn way - Start')
    accuracy_score = metrics.accuracy_score(actual_res, pred)
    print("accuracy_score = %0.3f" % (accuracy_score))
    f1_score = metrics.f1_score(actual_res, pred)
    print("f1_score = %0.3f" % (f1_score))
    precision_score = metrics.precision_score(actual_res, pred)
    print("precision_score = %0.3f" % (precision_score))
    recall_score = metrics.recall_score(actual_res, pred)
    print("recall_score = %0.3f" % (recall_score))
    print('generate the performance metrics for the prediction - sk-learn way - End')
    # generate the performance metrics for the prediction - sk-learn way - End

    # compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precision, and Max Precition @ k
    # threshold argument determines normal what recall values to calculate precision at
    # ### as per the author email on 30-June-2022: "In the current code I'm working on (which utilizes that file), I just use a threshold
    # list of [0.03, 1.0], as I don't use the other numbers currently.  With that, I just grab the first number from
    #  the max precision list (Precision at 3% recall) and last number from the average precision list (average precision at 100% recall)."
    thresholds = [0.03, 1.0]
    # thresholds = [0.03]
    score_dict = test_util.calcScores(actual_res, pred_prob_1_arr, thresholds)
    print('\n ####### score_dict:\n' + str(score_dict) + '\n #####################')
    # save score dictionary
    print('saving the score_dict calculated from the prediction result as a pkl file')
    score_dict_pkl_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random20_' + test_pkl_name_without_ext + '_score.pkl')
    joblib.dump(value=score_dict, filename=score_dict_pkl_name_with_loc, compress=3)
    # create score df and save it
    print('createing and saving the score_df created from the score_dict')
    score_df = pd.DataFrame({'dataset': ['Rand20'], 'testset': [test_pkl_name_without_ext]
                            , 'ACC': [score_dict['ACC']], 'AUC': [score_dict['AUC']], 'Prec': [score_dict['Prec']], 'Recall': [score_dict['Recall']]
                            , 'Thresholds': [score_dict['Thresholds']], 'Max Precision': [score_dict['Max Precision']]
                            , 'Avg Precision': [score_dict['Avg Precision']]
                            })
    score_df_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random20_' + test_pkl_name_without_ext + '_score.csv')
    score_df.to_csv(score_df_name_with_loc, index=False)
    print('#### inside the test_model() method - End')

# this method is used for the calculation of the overall score based on the average of the individual scores
def calcOverallScore(root_path='./'):
    print('\n#### inside the calcOverallScore() method - Start\n')
    # two iterations are needed for the 10% known interaction test data (Test1_) and
    # 0.3% known interaction test data (Test2_)
    print('two iterations are needed for the 10% known interaction test data (Test1_) and 0.3% known interaction test data (Test2_)')
    for test_index in range(1, 3):
        print('\n test_index: ' + str(test_index))
        no_of_test_files = 5
        con_score_df = None
        for ind in range(0, no_of_test_files):
        # for ind in range(4, 5):
            indiv_score_csv_file_nm = 'Random20_Test' + str(test_index) + '_' + str(ind) + '_score.csv'
            print('indiv_score_csv_file_nm: ' + indiv_score_csv_file_nm)
            indiv_score_df = pd.read_csv(os.path.join(root_path, TEST_RES_DIR, indiv_score_csv_file_nm))
            # store the indiv_score_df into con_score_df
            con_score_df = indiv_score_df if con_score_df is None else pd.concat([con_score_df, indiv_score_df], axis=0, sort=False)
        # end of for loop: for ind in range(0, no_of_test_files):

        # calculate avg_Prec
        # Precision at 3% recall indicates the 0th number from the max precision list ('Max Precision') when the threshold is [0.03, 1.0].
        # So the avg_Prec will be the average of all such 'Precision at 3% recall' values.
        con_Prec_lst = []
        max_p_arr = con_score_df['Max Precision'].to_numpy()
        # iterate over the max_p_arr whose each element is a string representation of the list of max-precisions
        for indiv_Max_Precision_str in max_p_arr:
            # indiv_Max_Precision_str is a string representation of the list of precisions
            indiv_Max_Precision_str = indiv_Max_Precision_str.replace('[', '').replace(']', '')  # remove '[ and ']'
            indiv_Max_Precision_lst = [float(prec) for prec in indiv_Max_Precision_str.split(',')]
            indiv_Prec_val = indiv_Max_Precision_lst[0]  # 0th number for 3% recall
            con_Prec_lst.append(indiv_Prec_val)
        avg_Prec = np.asarray(con_Prec_lst).mean()
        # calculate the avg_Avg_P
        # Average Precision at 100% recall indicates the last number in the avg precision list ('Avg Precision') when the threshold is [0.03, 1.0].
        # So the avg_Avg_P will be the average of all such 'Average Precision at 100% recall' values.
        con_avg_p_lst = []
        avg_p_arr = con_score_df['Avg Precision'].to_numpy()
        # iterate over the avg_p_arr whose each element is a string representation of the list of precisions
        for indiv_Avg_P_str in avg_p_arr:
            # indiv_Avg_P_str is a string representation of the list of precisions
            indiv_Avg_P_str = indiv_Avg_P_str.replace('[', '').replace(']', '')  # remove '[ and ']'
            indiv_Avg_P_lst = [float(prec) for prec in indiv_Avg_P_str.split(',')]
            indiv_Avg_P_val = indiv_Avg_P_lst[-1]  # last number for 100% recall
            con_avg_p_lst.append(indiv_Avg_P_val)
        avg_Avg_P = np.asarray(con_avg_p_lst).mean()
        print('avg_Prec = ' + str(avg_Prec) + ' :: avg_Avg_P ' + str(avg_Avg_P))
        # save the average values as separate rows of the con_score_df
        con_score_df['avg_Prec'] = [avg_Prec] + [''] * (con_score_df.shape[0] - 1)
        con_score_df['avg_Avg_P'] = [avg_Avg_P] + [''] * (con_score_df.shape[0] - 1)
        # save con_score_df
        con_score_df.to_csv(os.path.join(root_path, TEST_RES_DIR, 'con_score_for_Test' + str(test_index) + '.csv'), index=False)
    # end of for loop: for test_index in range(1, 3):
    print('#### inside the calcOverallScore() method - End')


def start(root_path='./', train_pkl_name = 'train.pkl'):
    # two iterations are needed for the 10% known interaction test data (Test1_) and
    # 0.3% known interaction test data (Test2_)
    print('two iterations are needed for the 10% known interaction test data (Test1_) and 0.3% known interaction test data (Test2_)')
    for test_index in range(1, 3):
        test_pkl_name = train_pkl_name.replace('Train', 'Test' + str(test_index))
        # next call test_model() method
        test_model(root_path, train_pkl_name, test_pkl_name)


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(1, 5):
        train_pkl_name = 'Train_' + str(ind) + '.pkl'
        start(root_path, train_pkl_name)
    calcOverallScore(root_path)