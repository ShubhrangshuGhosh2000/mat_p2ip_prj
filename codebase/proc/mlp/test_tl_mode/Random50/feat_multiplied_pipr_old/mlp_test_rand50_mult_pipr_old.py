import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torch.nn import functional as F
import glob
from sklearn import metrics
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.preproc.feat_engg import feat_multiplied_pipr_old
from codebase.utils import test_util
from codebase.utils.custom_2d_feat_dataset import Custom2DfeatDataset
from codebase.proc.mlp.train.Random50.feat_multiplied_pipr_old.mlp_train_rand50_mult_pipr_old import MlpModelRand50MultPipr

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # this is needed for the lambda implementation
CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_multiplied_pipr'
TEST_RES_DIR = 'dataset/proc_data/mlp_data/test_data/Random50/feat_multiplied_pipr'

def load_final_ckpt_model(root_path='./', train_tsv_name = 'train.tsv'):
    print('#### inside the load_final_ckpt_model() method - Start')
    train_tsv_name_without_ext = train_tsv_name.replace('.tsv', '') 
    # create the final checkpoint path
    final_ckpt_dir = os.path.join(root_path, CHECKPOINTS_DIR, train_tsv_name_without_ext + '_results')
    final_ckpt_path = os.path.join(final_ckpt_dir, "MlpModelRand50MultPipr_" + train_tsv_name_without_ext + '*.ckpt')
    final_ckpt_file_name = glob.glob(final_ckpt_path, recursive=False)[0]
    print('final_ckpt_file_name: ' + str(final_ckpt_file_name))
    # load the model
    model = MlpModelRand50MultPipr.load_from_checkpoint(final_ckpt_file_name)
    # # automatically restores model, epoch, step, LR schedulers, apex, etc...
    # model = MlpModelRand50MultPipr()
    # trainer = pl.Trainer()
    # trainer.fit(model, ckpt_path=final_ckpt_file_name)
    # transfer the model to the proper device
    print('### dev: ' + str(dev))
    model = model.to(dev)
    print('#### inside the load_final_ckpt_model() method - End')
    return model

def prepare_test_data(root_path='./', train_tsv_name = 'train.tsv', test_tsv_name = 'test.tsv', model = None):
    print('#### inside the prepare_test_data() method - Start')
    # call the data preparation method for the training tsv file 
    two_d_feat_arr_1_test, two_d_feat_arr_2_test, label_arr_1d_test = feat_multiplied_pipr_old.data_prep_feat_multiplied_pipr(root_path, 
    dataset_type='Random50', tsv_file_nm=test_tsv_name)
    two_d_feat_arr_1_test, two_d_feat_arr_2_test, label_arr_1d_test = two_d_feat_arr_1_test.astype(np.float32),\
        two_d_feat_arr_2_test.astype(np.float32), label_arr_1d_test.astype(int)
    print('#### inside the prepare_test_data() method - End')
    return (two_d_feat_arr_1_test, two_d_feat_arr_2_test, label_arr_1d_test)


def test_model(root_path='./', train_tsv_name = 'train.tsv', test_tsv_name = 'test.tsv'):
    print('\n #### inside the test_model() method - Start\n')
    print('# #############################\n')
    print('train_tsv_name: ' + train_tsv_name + '\ntest_tsv_name: ' + test_tsv_name)
    print('\n# #############################')

    # load the final checkpointed model
    print('loading the final checkpointed model')
    model = load_final_ckpt_model(root_path, train_tsv_name)
    # prepare the test data
    print('\n preparing the test data')
    two_d_feat_arr_1_test, two_d_feat_arr_2_test, label_arr_1d_test = prepare_test_data(root_path, train_tsv_name, test_tsv_name, model)

    # create test data loader
    test_data = Custom2DfeatDataset(two_d_feat_arr_1_test, two_d_feat_arr_2_test, label_arr_1d_test)
    test_dataloader = DataLoader(test_data, batch_size=int(model.batch_size)
        , num_workers=os.cpu_count(), pin_memory= True if(torch.cuda.is_available()) else False)
    # perform the prediction
    print('\n performing the prediction')
    # disable the gradiant tracking by pytorch
    torch.set_grad_enabled(False)
    # set model in the evaluation mode
    model.eval()
    logits_lst = []
    for batch_idx, pred_batch in enumerate(test_dataloader):
        pred = model.predict_step(pred_batch, batch_idx)
        logits_lst.append(pred)
    # concatenate the logits_lst of tensors
    logits = torch.cat(logits_lst, dim=0)

    # find the predicted class (0 or 1)
    __, pred = torch.max(logits.data, 1)
    # find the prediction probability
    # logits contains the log of probability(i.e. softmax) in a 2d format where data-points are arranged
    # in rows and columns contain class-0 logit and class-1 logit. Its dimension is n x 2 where n is the
    # number of data points and there are 2 columns containing logits values for class-0 and class-1 respectively.
    prob_2d_arr = F.softmax(logits, dim=1)

    # create a df to store the prediction result
    print('creating a df to store the prediction result')
    test_tsv_input_path = os.path.join(root_path, 'dataset/orig_data/Human2021/BioGRID2021', 'Random50')
    test_tsv_df = pd.read_csv(os.path.join(test_tsv_input_path, test_tsv_name), header=None, sep='\t')
    actual_res = label_arr_1d_test
    pred = pred.cpu().detach().numpy()
    pred_prob_0_arr = prob_2d_arr[:, 0].cpu().detach().numpy()
    pred_prob_1_arr = prob_2d_arr[:, 1].cpu().detach().numpy()
    pred_result_df = pd.DataFrame({'prot_1': test_tsv_df[0], 'prot_2': test_tsv_df[1]
                                    , 'actual_res': actual_res
                                    , 'pred_res': pred
                                    , 'pred_prob_0': pred_prob_0_arr
                                    , 'pred_prob_1': pred_prob_1_arr
                                    })
    # save the pred_result_df
    test_tsv_name_without_ext = test_tsv_name.replace('.tsv', '') 
    file_nm_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random50_' + test_tsv_name_without_ext + '_pred_res.csv')
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
    print('saving the score_dict calculated from the prediction result as a tsv file')
    score_dict_tsv_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random50_' + test_tsv_name_without_ext + '_score.tsv')
    joblib.dump(value=score_dict, filename=score_dict_tsv_name_with_loc, compress=3)
    # create score df and save it
    print('createing and saving the score_df created from the score_dict')
    score_df = pd.DataFrame({'dataset': ['Rand50'], 'testset': [test_tsv_name_without_ext]
                            , 'ACC': [score_dict['ACC']], 'AUC': [score_dict['AUC']], 'Prec': [score_dict['Prec']], 'Recall': [score_dict['Recall']]
                            , 'Thresholds': [score_dict['Thresholds']], 'Max Precision': [score_dict['Max Precision']]
                            , 'Avg Precision': [score_dict['Avg Precision']]
                            })
    score_df_name_with_loc = os.path.join(root_path, TEST_RES_DIR
                                           , 'Random50_' + test_tsv_name_without_ext + '_score.csv')
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
    con_score_df['avg_ACC'] = [''] * con_score_df.shape[0]
    con_score_df.loc[0, 'avg_ACC'] = avg_ACC
    con_score_df['avg_AUC'] = [''] * con_score_df.shape[0]
    con_score_df.loc[0, 'avg_AUC'] = avg_AUC
    # save con_score_df
    con_score_df.to_csv(os.path.join(root_path, TEST_RES_DIR, 'con_score_df.csv'), index=False)
    print('#### inside the calcOverallScore() method - End')


def start(root_path='./', train_tsv_name = 'train.tsv'):
    test_tsv_name = train_tsv_name.replace('Train', 'Test')
    # next call test_model() method
    test_model(root_path, train_tsv_name, test_tsv_name)


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
        train_tsv_name = 'Train_' + str(ind) + '.tsv'
        start(root_path, train_tsv_name)
    calcOverallScore(root_path)