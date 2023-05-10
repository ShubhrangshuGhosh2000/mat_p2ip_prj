import os
import sys
from pathlib import Path

import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)


CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side'

def start(root_path='./'):
    # Limit the training time, in second
    time_limit = 1800

    csv_file_path = os.path.join(root_path, CHECKPOINTS_DIR, 'scaled_reduced_trn_tst_csv')
    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(0, 1):
        result_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'autogluon_train', 'Train_' + str(ind) + '_results')
        # create the result_dir if it does not already exist
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # create train csv file name
        train_csv_file_name = os.path.join(csv_file_path, 'train_' + str(ind) + '.csv')
        # Load training data from a CSV file into an AutoGluon Dataset object.
        # This object is essentially equivalent to a Pandas DataFrame and the same methods can be applied to both.
        # Note: This train data is already scaled and dimensionally reduced
        train_data = TabularDataset(train_csv_file_name)
        train_data.head()

        # Now use AutoGluon to train multiple models in LIGHTER mode
        print('\n #################### Now using AutoGluon to train multiple models in LIGHTER mode - START ####################')
        predictor = TabularPredictor(label='target', verbosity=3, eval_metric='accuracy', path=result_dir).fit(train_data
                    , time_limit=time_limit  # Limit the training time, in second
                    , ag_args_fit={'num_gpus': 1}  # for GPU support
                    , presets='best_quality'  # Better model ensemble for a better accuracy, 
                                              # but longer training time. All available options.
                    , tuning_data=None  # # Use a separate dataset to tune models.
                    , hyperparameters='very_light'  # # Explore less models. You can fully control the model search space. All available options.
                    , excluded_model_types=['KNN']  # # Ignore some models. 'NN_TORCH'
                    , refit_full=True  # Whether to retrain all models on all of the data (training + validation) after the normal training procedure. 
                    )
        print('\n #################### Now using AutoGluon to train multiple models in LIGHTER mode - END ####################')

        # # Now use AutoGluon to train multiple models in LIGHTER mode
        # print('\n #################### Now using AutoGluon to train multiple models in HEAVY mode - START ####################')
        # nn_options = {  # specifies non-default hyperparameter values for neural network models
        #     'num_epochs': 10,  # number of training epochs (controls training time of NN models)
        #     'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True),  # learning rate used in training (real-valued hyperparameter searched on log-scale)
        #     'activation': ag.space.Categorical('relu', 'softrelu', 'tanh'),  # activation function used in NN (categorical hyperparameter, default = first entry)
        #     'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1),  # dropout probability (real-valued hyperparameter)
        # }

        # gbm_options = {  # specifies non-default hyperparameter values for lightGBM gradient boosted trees
        #     'num_boost_round': 100,  # number of boosting rounds (controls training time of GBM models)
        #     'num_leaves': ag.space.Int(lower=26, upper=66, default=36),  # number of leaves in trees (integer hyperparameter)
        # }

        # hyperparameters = {  # hyperparameters of each model type
        #                 'GBM': gbm_options,
        #                 'NN_TORCH': nn_options,  # NOTE: comment this line out if you get errors on Mac OSX
        #                 }  # When these keys are missing from hyperparameters dict, no models of that type are trained

        # num_trials = 5  # try at most 5 different hyperparameter configurations for each type of model
        # search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

        # hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        #     'num_trials': num_trials,
        #     'scheduler' : 'local',
        #     'searcher': search_strategy,
        # }

        # predictor = TabularPredictor(label='target', verbosity=3, eval_metric='accuracy', path=result_dir).fit(train_data
        #             , time_limit=time_limit
        #             , hyperparameters=hyperparameters
        #             , hyperparameter_tune_kwargs=hyperparameter_tune_kwargs
        #             )
        # print('\n #################### Now using AutoGluon to train multiple models in HEAVY mode - END ####################')

        # Understand the contribution of each model
        predictor.leaderboard()
        # Understand more about the trained models
        print('\n### predictor fit_summary ###\n')
        predictor.fit_summary(show_plot=True)
        print('\n### the predictor is save in the path: ' + str(predictor.path) + ' ###')
        # get the dictionary of original model name -> refit full model name
        print('\n ### model_full_dict: ' + str(predictor.get_model_full_dict()))
        # get the string model name of the best model by validation score that can infer.
        print('\n ### the best model based on the validation score : ' + str(predictor.get_model_best()))
    # end of for loop
    print('\n ##################### END OF THE AUTOGLUON PROCESS ######################')


if __name__ == '__main__':
    root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    start(root_path)
