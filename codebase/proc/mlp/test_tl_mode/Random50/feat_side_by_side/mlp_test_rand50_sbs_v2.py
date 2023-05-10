import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import umap
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torch.nn import functional as F
import glob
import pandas as pd
from sklearn import metrics

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.utils import test_util
from codebase.proc.mlp.train.Random50.feat_side_by_side.mlp_train_rand50_sbs_fixed_lr_train_v2 import MlpModelRand50Sbs_fixedLrTrain

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side'
TEST_RES_DIR = 'dataset/proc_data/mlp_data/test_data/Random50/feat_side_by_side'

def load_final_ckpt_model(root_path='./', train_pkl_name = 'train.pkl'):
    print('#### inside the load_final_ckpt_model() method - Start')
    # create the final checkpoint path
    final_ckpt_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'fixed_lr_train_results')
    train_pkl_name_without_ext = train_pkl_name.replace('.pkl', '') 
    final_ckpt_path = os.path.join(final_ckpt_dir, "MlpModelRand50Sbs_fixedLrTrain_" + str(train_pkl_name_without_ext) + '*.ckpt')
    final_ckpt_file_name = glob.glob(final_ckpt_path, recursive=False)[0]
    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = MlpModelRand50Sbs_fixedLrTrain.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = MlpModelRand50Sbs_fixedLrTrain()
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)
    print('#### inside the load_final_ckpt_model() method - End')
    return model


def prepare_test_data(root_path='./', train_pkl_name = 'train.pkl', test_pkl_name = 'test.pkl', model = None):
    print('#### inside the prepare_test_data() method - Start')
    test_arr_2d = None
    # check whether the part of preprocessing (upto dim-reduction) is already done and saved as a pkl file
    print('checking whether the part of preprocessing (upto dim-reduction) is already done and saved as a pkl file')
    test_pkl_name_without_ext = test_pkl_name.replace('.pkl', '') 
    test_part_preproc_pkl_path = os.path.join(root_path,
    TEST_RES_DIR, test_pkl_name_without_ext + '_part_preproc_final.pkl')
    print('test_part_preproc_pkl_path: ' + str(test_part_preproc_pkl_path))
    if os.path.exists(test_part_preproc_pkl_path):
        # as the partial test preproc file already exists, skipping upto dim-reduction step
        print('\n##### As the partial test preproc file already exists, skipping upto dim-reduction step...')
        # load already saved ptest reproc file
        test_part_preproc_dict = joblib.load(test_part_preproc_pkl_path)
        test_arr_2d = test_part_preproc_dict['test_arr_2d']
        X_test_scaled_reduced = test_part_preproc_dict['X_test_scaled_reduced']
        y_test_arr = test_part_preproc_dict['y_test_arr']
    else:
        # start data preparation from the beginning as no partial preproc file exists
        print('\n##### starting test data preparation from the beginning as no partial test preproc file exists...')
        test_data_path = os.path.join(root_path,
        'dataset/preproc_data/human_2021/Random50/feat_side_by_side', test_pkl_name)
        print('test_data_path: ' + str(test_data_path))
        # load the pkl file
        print('loading the pkl file')
        test_lst = joblib.load(test_data_path)
        # test_lst is a lst of 1d arrays; now convert it into a 2d array
        test_arr_2d = np.vstack(test_lst)
        # retrieve the scaled and dimensionally reduced test data, already saved during the training preprocessing using the 
        # same scaler and reducer as used for the training data
        train_pkl_name_without_ext = train_pkl_name.replace('.pkl', '') 
        train_part_preproc_pkl_path = os.path.join(root_path,
        CHECKPOINTS_DIR, train_pkl_name_without_ext + '_part_preproc_dim_' + str(model.reduc_dim) + '.pkl')
        print('train_part_preproc_pkl_path: ' + str(train_part_preproc_pkl_path))
        # load already saved preproc file
        train_part_preproc_dict = joblib.load(train_part_preproc_pkl_path)
        X_test_scaled_reduced = train_part_preproc_dict['X_test_scaled_reduced']
        y_test_arr = train_part_preproc_dict['y_test_arr']
        # save the result upto the dim-reduction step as the partial preproc file so that it can be reused 
        print('saving the result upto the dim-reduction step as the partial preproc file so that it can be reused')
        joblib.dump(value={'test_arr_2d': test_arr_2d, 'X_test_scaled_reduced': X_test_scaled_reduced, 'y_test_arr': y_test_arr}
                    , filename=test_part_preproc_pkl_path
                    , compress=3)
    X_test = X_test_scaled_reduced
    y_test = y_test_arr

    # tranform the 2d numpy arrays to the torch tensors
    print('transforming the 2d numpy arrays to the torch tensors')
    X_test = X_test.astype(np.float32)
    X_test = torch.from_numpy(X_test)
    # transform the 1d numpy array of floats to the array of int as the target label is integer
    # no need for np.round() here as the floats are either 0.0 or 1.0
    y_test = y_test.astype(int)
    print('#### inside the prepare_test_data() method - End')
    return (test_arr_2d, X_test, y_test)


def test_model(root_path='./', train_pkl_name = 'train.pkl', test_pkl_name = 'test.pkl'):
    print('\n #### inside the test_model() method - Start\n')
    print('# #############################\n')
    print('train_pkl_name: ' + train_pkl_name + '\ntest_pkl_name: ' + test_pkl_name)
    print('\n# #############################')

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
                                           , 'Random50_' + test_pkl_name_without_ext + '_pred_res.csv')
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
    # threshold argument determinesnormal what recall values to calculate precision at
    thresholds = [0.03, 1.0]
    score_dict = test_util.calcScores(actual_res, pred_prob_1_arr, thresholds)
    print('\n ####### score_dict:\n' + str(score_dict) + '\n #####################')
    # save score dictionary
    print('saving the score_dict calculated from the prediction result as a pkl file')
    score_dict_pkl_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random50_' + test_pkl_name_without_ext + '_score.pkl')
    joblib.dump(value=score_dict, filename=score_dict_pkl_name_with_loc, compress=3)
    # create score df and save it
    print('createing and saving the score_df created from the score_dict')
    score_df = pd.DataFrame({'dataset': ['Rand50'], 'testset': [test_pkl_name_without_ext]
                            , 'ACC': [score_dict['ACC']], 'AUC': [score_dict['AUC']], 'Prec': [score_dict['Prec']], 'Recall': [score_dict['Recall']]
                            , 'Thresholds': [score_dict['Thresholds']], 'Max Precision': [score_dict['Max Precision']]
                            , 'Avg Precision': [score_dict['Avg Precision']]
                            })
    score_df_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random50_' + test_pkl_name_without_ext + '_score.csv')
    score_df.to_csv(score_df_name_with_loc, index=False)
    print('#### inside the test_model() method - End')


def calcOverallScore(root_path='./'):
    print('#### inside the calcOverallScore() method - Start')
    con_score_df = None
    no_of_test_files = 5
    for ind in range(0, no_of_test_files):
    # for ind in range(4, 5):
        indiv_score_csv_file_nm = 'Random50_Test_' + str(ind) + '_score.csv'
        print('indiv_score_csv_file_nm: ' + indiv_score_csv_file_nm)
        indiv_score_df = pd.read_csv(os.path.join(root_path, TEST_RES_DIR, indiv_score_csv_file_nm))
        # store the indiv_score_df into con_score_df
        con_score_df = indiv_score_df if con_score_df is None else pd.concat([con_score_df, indiv_score_df], axis=0, sort=False)
    # end of for loop
    # find average ACC and AUC val
    avg_ACC = con_score_df['ACC'].mean()
    avg_AUC = con_score_df['AUC'].mean()
    print('avg_ACC = ' + str(avg_ACC) + ' :: avg_AUC ' + str(avg_AUC))
    # save the average values as separate rows of the con_score_df
    con_score_df['avg_ACC'] = [avg_ACC] + [''] * (con_score_df.shape[0] - 1)
    con_score_df['avg_AUC'] = [avg_AUC] + [''] * (con_score_df.shape[0] - 1)
    # save con_score_df
    con_score_df.to_csv(os.path.join(root_path, TEST_RES_DIR, 'con_score_df.csv'), index=False)
    print('#### inside the calcOverallScore() method - End')


def start(root_path='./', train_pkl_name = 'train.pkl'):
    test_pkl_name = train_pkl_name.replace('Train', 'Test')
    # next call test_model() method
    test_model(root_path, train_pkl_name, test_pkl_name)


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(4, 5):
        train_pkl_name = 'Train_' + str(ind) + '.pkl'
        # start(root_path, train_pkl_name)
    calcOverallScore(root_path)