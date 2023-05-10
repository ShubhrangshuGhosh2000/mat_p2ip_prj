import glob
import os
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.proc.mlp.train_tl_mode.Random50.feat_side_by_side.mlp_train_rand50_sbs_lr_finder_v3_pca import MlpModelRand50Sbs_lrFinder
from utils import dl_reproducible_result_util

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side'

# Multilayer Perceptron (MLP) Model for the Side-By-Side(sbs) features using Randim50 training sets
class MlpModelRand50Sbs_fixedLrTrain(MlpModelRand50Sbs_lrFinder):

    def __init__(self, config = None, root_path='./', result_dir='./', pkl_name = 'Train.pkl'):
        super(MlpModelRand50Sbs_fixedLrTrain, self).__init__(config, root_path, result_dir, pkl_name)

def train_model(config = None, root_path='./', result_dir='./', pkl_name = 'train.pkl', num_epochs = 10):
    print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random number generators in: pytorch, numpy, python.random 
    pl.seed_everything(seed=456, workers=True)
    # instantiate the model class
    model = MlpModelRand50Sbs_fixedLrTrain(config, root_path, result_dir, pkl_name)
    pkl_name_without_ext = pkl_name.replace('.pkl', '') 
    ckpt_file_nm_1st_part = "MlpModelRand50Sbs_fixedLrTrain_" + pkl_name_without_ext + '-' + str(config["layer_1_size"]) + "-" + str(config["layer_2_size"]) + "-" + str(config["layer_3_size"]) \
    + "-batch" + str(config["batch_size"]) + "-dim" + str(config["reduc_dim"]) + "-lr{:0.4f}-".format(config["lr"])
    ckpt_file_nm_format = ckpt_file_nm_1st_part + "-epoch{epoch:02d}-val_accuracy{train/val_accuracy:.3f}"
    
    # define the checkpoint_callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        dirpath=result_dir
                        , filename=ckpt_file_nm_format
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        # , monitor='train/val_loss', mode='min'
                        , monitor='train/val_accuracy', mode='max'
                        , every_n_epochs = 1, verbose=True)

     # define the early_stop_callback
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        # monitor="train/val_loss"
        monitor="train/val_accuracy", mode="max"
        , min_delta=0.00, patience=5000, verbose=True)  # monitor="val_acc" can be checked

    # # instantiate neptune logger
    # logger = NeptuneLogger(project="sg-neptune/only-seq-prj-v1"
    # , api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzM4ODM3Mi1iZDZjLTQ1MDUtYmYxZi00Mzg3OTA4YTc1NTUifQ=="
    # , name="", prefix='', tags=["training-random50-sbs"]
    # )

    # instantiate tensorboard logger
    tb_dir = os.path.join(result_dir, 'tb_logs')
    # tb_dir = tune.get_trial_dir()
    logger = TensorBoardLogger(save_dir=tb_dir, name='', version=".", prefix='', log_graph=True)

    resume_from_checkpoint = None
    # Get a list of all the file paths that ends with .ckpt in specified directory
    ckpt_fileList = glob.glob(os.path.join(result_dir, ckpt_file_nm_1st_part + '*.ckpt'), recursive=False)
    if(len(ckpt_fileList) != 0):  # ckpt file exists
        resume_from_checkpoint = ckpt_fileList[0]

    # instantiate a trainer
    trainer = pl.Trainer(fast_dev_run=False, max_epochs=num_epochs
                        , deterministic = True
                        , logger = logger
                        # , logger = False
                        , callbacks=[checkpoint_callback, early_stop_callback] # can also use checkpoint_callback, early_stop_callback
                        , gpus = -1 if(torch.cuda.is_available()) else 0
                        , precision = 16
                        # , resume_from_checkpoint = resume_from_checkpoint  # deprecated, use 'ckpt_path' in fit() method instead
                        , auto_lr_find = None
                        , enable_progress_bar = False
                        , enable_model_summary = True)

    print('#### before calling trainer.fit(model) method')
    trainer.fit(model, ckpt_path = resume_from_checkpoint)
    print('#### after calling trainer.fit(model) method')
    # print('#### inside the train_model() method - End')


def start(root_path='./'):
    config = None
    num_epochs = 100

    # retrieve the con_best_configs_df.csv
    con_best_configs_df = pd.read_csv(os.path.join(root_path, CHECKPOINTS_DIR, 'con_best_configs_df.csv'))
    result_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'fixed_lr_train_results')
    # create the result_dir if it does not already exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    no_of_train_files = 5
    for ind in range(0, no_of_train_files):
    # for ind in range(4, 5):
        pkl_name = 'Train_' + str(ind) + '.pkl'
        print('\n# ############################# training pkl_name: ' + pkl_name + '\n')        
        config = {
        "layer_1_size": con_best_configs_df.loc[0, 'layer_1_size']
        , "layer_2_size": con_best_configs_df.loc[0, 'layer_2_size']
        , "layer_3_size": con_best_configs_df.loc[0, 'layer_3_size']
        , "batch_size": con_best_configs_df.loc[0, 'batch_size']
        , "reduc_dim": con_best_configs_df.loc[0, 'reduc_dim']
        , "lr": con_best_configs_df.loc[0, 'min_lr']
        }
        print('\n ##### config #####\n ' + str(config))
        train_model(config, root_path, result_dir, pkl_name, num_epochs)
    # end of for loop
    print('\n ##################### END OF THE FIXED lr TRAINING PROCESS ######################')


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    start(root_path)
