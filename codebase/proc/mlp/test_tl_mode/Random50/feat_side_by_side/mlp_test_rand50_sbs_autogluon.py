import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.utils import test_util

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side'
TEST_RES_DIR = 'dataset/proc_data/mlp_data/test_data/Random50/feat_side_by_side'

def test_model(root_path='./', train_pkl_name = 'train.pkl', test_pkl_name = 'test.pkl'):
    print('\n #### inside the test_model() method - Start\n')
    print('# #############################\n')
    print('train_pkl_name: ' + train_pkl_name + '\ntest_pkl_name: ' + test_pkl_name)
    print('\n# #############################')
    test_result_dir = os.path.join(root_path, TEST_RES_DIR, 'autogluon_test')
    # create the test_result_dir if it does not already exist
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    # Load saved model(s) from disk
    print('\nloading the saved model(s) from disk')
    train_pkl_name_without_ext = train_pkl_name.replace('.pkl', '') 
    trained_model_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'autogluon_train', train_pkl_name_without_ext + '_results')
    predictor = TabularPredictor.load(trained_model_dir, verbosity=4)

    # load the test data
    print('l\noading the test data (already scaled and dimensionally reduced)')
    csv_file_path = os.path.join(root_path, CHECKPOINTS_DIR, 'scaled_reduced_trn_tst_csv')
    test_pkl_name_without_ext = test_pkl_name.replace('.pkl', '') 
    # create test csv file name
    test_csv_file_name = os.path.join(csv_file_path, str.lower(test_pkl_name_without_ext) + '.csv')
    # Load test data from a CSV file into an AutoGluon Dataset object.
    # This object is essentially equivalent to a Pandas DataFrame and the same methods can be applied to both.
    # Note: This test data is already scaled and dimensionally reduced
    test_data = TabularDataset(test_csv_file_name)

    # perform the prediction
    print('\n ### performing the prediction\n')
    label = 'target'
    actual_res = test_data[label]  # values to predict
    test_data_nolab = test_data.drop(columns=[label])  # delete the label column

    # Predict for each row
    # y_pred = predictor.predict(test_data_nolab)
    # Predict using a specific model (here the best model can be used)
    best_model_name = predictor.get_model_best()
    y_pred = predictor.predict(test_data_nolab, model=best_model_name)

    # Return the class probabilities for classification
    pred_probs = predictor.predict_proba(test_data_nolab, as_pandas=True, as_multiclass=True)
    pred_prob_0_arr = pred_probs[predictor.class_labels[0]].to_numpy()
    pred_prob_1_arr = pred_probs[predictor.class_labels[1]].to_numpy()

    # create a df to store the prediction result
    print('\ncreating a df to store the prediction result')
    pred_result_df = pd.DataFrame({ 'actual_res': actual_res
                                    , 'pred_res': y_pred
                                    , 'pred_prob_0': pred_prob_0_arr
                                    , 'pred_prob_1': pred_prob_1_arr
                                    })
    # save the pred_result_df
    file_nm_with_loc = os.path.join(test_result_dir, 'Random50_' + test_pkl_name_without_ext + '_pred_res.csv')
    pred_result_df.to_csv(file_nm_with_loc, index=False)

    print('\n ### evaluating the model using the predictions')
    # perf = predictor.evaluate_predictions(y_true=actual_res, y_pred=y_pred, auxiliary_metrics=True)

    # compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precision, and Max Precition @ k
    # threshold argument determinesnormal what recall values to calculate precision at
    thresholds = [0.03, 1.0]
    score_dict = test_util.calcScores(actual_res, pred_prob_1_arr, thresholds)
    print('\n ####### score_dict:\n' + str(score_dict) + '\n #####################')
    # save score dictionary
    print('saving the score_dict calculated from the prediction result as a pkl file')
    score_dict_pkl_name_with_loc = os.path.join(test_result_dir, 'Random50_' + test_pkl_name_without_ext + '_score.pkl')
    joblib.dump(value=score_dict, filename=score_dict_pkl_name_with_loc, compress=3)
    # create score df and save it
    print('createing and saving the score_df created from the score_dict')
    score_df = pd.DataFrame({'dataset': ['Rand50'], 'testset': [test_pkl_name_without_ext]
                            , 'ACC': [score_dict['ACC']], 'AUC': [score_dict['AUC']], 'Prec': [score_dict['Prec']], 'Recall': [score_dict['Recall']]
                            , 'Thresholds': [score_dict['Thresholds']], 'Max Precision': [score_dict['Max Precision']]
                            , 'Avg Precision': [score_dict['Avg Precision']]
                            })
    score_df_name_with_loc = os.path.join(test_result_dir, 'Random50_' + test_pkl_name_without_ext + '_score.csv')
    score_df.to_csv(score_df_name_with_loc, index=False)
    print('#### inside the test_model() method - End')


def calcOverallScore(root_path='./'):
    print('#### inside the calcOverallScore() method - Start')
    con_score_df = None
    no_of_test_files = 5
    for ind in range(0, no_of_test_files):
    # for ind in range(0, 1):
        indiv_score_csv_file_nm = 'Random50_Test_' + str(ind) + '_score.csv'
        print('indiv_score_csv_file_nm: ' + indiv_score_csv_file_nm)
        indiv_score_df = pd.read_csv(os.path.join(root_path, TEST_RES_DIR, 'autogluon_test', indiv_score_csv_file_nm))
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
    con_score_df.to_csv(os.path.join(root_path, TEST_RES_DIR, 'autogluon_test', 'con_score_df.csv'), index=False)
    print('#### inside the calcOverallScore() method - End')


def start(root_path='./', train_pkl_name = 'train.pkl'):
    test_pkl_name = train_pkl_name.replace('Train', 'Test')
    # next call test_model() method
    test_model(root_path, train_pkl_name, test_pkl_name)


if __name__ == '__main__':
    root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(0, 1):
        train_pkl_name = 'Train_' + str(ind) + '.pkl'
        start(root_path, train_pkl_name)
    calcOverallScore(root_path)
