import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from sklearn import metrics, preprocessing, model_selection
from torch.nn import functional as F
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[5]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils.custom_dataset import CustomDataset
from utils import test_util, dl_reproducible_result_util

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data_man_mode/HeldOut20/feat_side_by_side'
thresholds = [0.03, 1.0]

# Multilayer Perceptron (MLP) Model for the Side-By-Side(sbs) features using HeldOut20 training sets
class MlpModelHeld20Sbs_lrFinder_ManMode(pl.LightningModule):
# ################################## VERY IMPORTANT NOTES -START #######################
# Although the initial idea was to find the best lr value for each individual model and then
# find the minimum of those lr values and finally use that minimum lr value to retrain all the models. But 
# now that plan is dropped and lr is kept to the fixed value for all the models from the very beginning. But
# the previous code-structure is kept as it is because it is not known whether the previous plan would be
# applicable again.
# ################################## VERY IMPORTANT NOTES -END #######################

    def __init__(self, config = None, root_path='./', result_dir='./', pkl_name = 'Train.pkl'):
        super(MlpModelHeld20Sbs_lrFinder_ManMode, self).__init__()
        self.root_path = root_path
        self.result_dir = result_dir
        self.pkl_name = pkl_name

        self.layer_1_size = config["layer_1_size"]
        self.layer_2_size = config["layer_2_size"]
        self.layer_3_size = config["layer_3_size"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.reduc_dim = config["reduc_dim"]

        # Defining the model architecture
        # for side-by-side (sbs), orginal feature vector length is 2048 (1024 for 
        # each interacting protein) 
        self.layer_1 = torch.nn.Linear(self.reduc_dim, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, self.layer_3_size)
        self.layer_4 = torch.nn.Linear(self.layer_3_size, 2)

        # save __init__ arguments to hparams attribute.
        self.save_hyperparameters()


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # print('#### inside the forward() method')
        batch_size, feat_vec_size = x.size()
        # (b, 2048) -> (b, 2048)
        x = x.view(batch_size, -1)
        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)
        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)
        # layer 3
        x = self.layer_3(x)
        x = torch.relu(x)
        # layer 4
        x = self.layer_4(x)
        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    # normal prec calculation
    def prec(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        # convert cuda type tensor to numpy array
        labels = labels.cpu().detach().numpy() 
        predicted = predicted.cpu().detach().numpy()
        prec = metrics.precision_score(labels, predicted)
        return torch.tensor(prec)

    # special prec calculation
    def prec_spl(self, logits, labels):
        # find the prediction probability
        # logits contains the log of probability(i.e. softmax) in a 2d format where data-points are arranged
        # in rows and columns contain class-0 logit and class-1 logit. Its dimension is n x 2 where n is the
        # number of data points and there are 2 columns containing logits values for class-0 and class-1 respectively.
        prob_2d_arr = F.softmax(logits, dim=1)
        pred_prob_1_arr = prob_2d_arr[:, 1]
        # convert cuda type tensor to numpy array
        labels = labels.cpu().detach().numpy() 
        pred_prob_1_arr = pred_prob_1_arr.cpu().detach().numpy()
        score_dict = test_util.calcScores(labels, pred_prob_1_arr, thresholds)
        prec = score_dict['Prec']
        return torch.tensor(prec)

    def training_step(self, train_batch, batch_idx):
        # REQUIRED
        # training_step defined the train loop.
        # It is independent of forward
        # print('#### inside the training_step() method')
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        prec = self.prec(logits, y)
        self.log("train/train_loss", loss)
        self.log("train/train_prec", prec)
        return {'loss': loss, 'prec': prec}

    def validation_step(self, val_batch, batch_idx):
        # OPTIONAL
        # print('#### inside the validation_step() method')
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        prec = self.prec(logits, y)
        return {"val_loss": loss, "val_prec": prec}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 

        print('#### inside the validation_epoch_end() method')
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_prec = torch.stack([x["val_prec"] for x in outputs]).mean()
        self.log("train/val_loss", avg_loss)  # ## this is important for model checkpointing
        self.log("train/val_prec", avg_prec)
        print('train/val_prec: ' + str(avg_prec.numpy()))
        return {'val_loss': avg_loss, "val_prec": avg_prec}

    def configure_optimizers(self):
        # REQUIRED
        # print('#### inside the configure_optimizers() method')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def prepare_data(self):
        # Even in multi-GPU training, this method is called only from a single GPU. 
        # So this method ideal for download, stratification etc.
        # print('#### inside the prepare_data() method -Start')
        # extract the manual features for the training data
        print('\nExtracting the manual features for the training data...')
        manual_training_data_path = os.path.join(self.root_path, 'dataset/preproc_data/human_2021_manual/HeldOut20/feat_side_by_side', self.pkl_name)
        print('manual_training_data_path: ' + str(manual_training_data_path))
        # load the train pkl file
        train_lst_manual = joblib.load(manual_training_data_path)
        # train_lst_manual is a lst of 1d arrays; now convert it into a 2d array
        train_arr_2d_manual = np.vstack(train_lst_manual)
        # next perform column filtering and column rearranging so that the feature columns come first and then
        # the target column (in train_arr_2d, the target column is in the 2th column index and features are started
        # from 3th column index onwards)
        train_arr_manual = train_arr_2d_manual[:, list(range(3, train_arr_2d_manual.shape[1])) + [2]]
        X_train_arr_manual = train_arr_manual[:, range(0, train_arr_manual.shape[1] -1)]  # excluding the target column
        y_train_arr_manual = train_arr_manual[:, -1]  # the last column i.e. target column
        print('X_train_arr_manual.shape: ' + str(X_train_arr_manual.shape))  # (, 3708)

        # z-normalize X_train_arr_manual feature(column) wise
        scaler = preprocessing.StandardScaler()
        X_train_scaled_reduced = scaler.fit_transform(X_train_arr_manual)
        y_train_arr = y_train_arr_manual
        print('X_train_scaled_reduced.shape: ' + str(X_train_scaled_reduced.shape))  # (, 3708)
        # save the scaler for the future use
        self.scaler = scaler
        self.reduc_dim = X_train_scaled_reduced.shape[1]
        # ######################### TEMP CODE -START #########################
        # val_lst_tl = []
        # val_lst_manual = []
        # for i in range(1, 3):
        #     val_pkl_name_manual = self.pkl_name.replace('Train', 'Test' + str(i))
        #     val_data_path_manual = os.path.join(self.root_path, 'dataset/preproc_data/human_2021_manual/HeldOut20/feat_side_by_side', val_pkl_name_manual)
        #     val_lst_manual += joblib.load(val_data_path_manual)
        # val_arr_2d_manual = np.vstack(val_lst_manual)
        # val_arr_manual = val_arr_2d_manual[:, list(range(3, val_arr_2d_manual.shape[1])) + [2]]
        # X_val_arr_manual = val_arr_manual[:, range(0, val_arr_manual.shape[1] -1)]
        # y_val_arr_manual = val_arr_manual[:, -1]

        # X_val_scaled_reduced = scaler.transform(X_val_arr_manual)
        # y_val_arr = y_val_arr_manual
        # ######################### TEMP CODE -END #########################
        # splits the data into training and validation sets
        print('splitting the data into training and validation sets')
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_train_scaled_reduced, y_train_arr, test_size=0.05, random_state=456, stratify=y_train_arr)
        # # TEMP CODE
        # X_train, X_val, y_train, y_val = X_train_scaled_reduced, X_val_scaled_reduced, y_train_arr, y_val_arr
        # tranform the 2d numpy arrays to the torch tensors
        print('transforming the 2d numpy arrays to the torch tensors')
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        X_train = torch.from_numpy(X_train)
        X_val = torch.from_numpy(X_val)
        # transform the 1d numpy array of floats to the array of int as the target label is integer
        # no need for np.round() here as the floats are either 0.0 or 1.0
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        # create the custom torch dataset to be used by the torch dataloader
        print('creating the custom torch dataset to be used by the torch dataloader')
        self.train_data = CustomDataset(X_train, y_train)
        self.val_data = CustomDataset(X_val, y_val)
        # print('#### inside the prepare_data() method -End')

    def train_dataloader(self):
        print('#### inside the train_dataloader() method')
        return DataLoader(self.train_data, batch_size=int(self.batch_size)
        , num_workers=os.cpu_count(), pin_memory= True if(torch.cuda.is_available()) else False
        , shuffle=True)

    def val_dataloader(self):
        print('#### inside the val_dataloader() method')
        return DataLoader(self.val_data, batch_size=int(self.batch_size)
        , num_workers=os.cpu_count(), pin_memory= True if(torch.cuda.is_available()) else False)

def train_model(config = None, root_path='./', result_dir='./', pkl_name = 'train.pkl', num_epochs = 10, lr_finder=False):
    print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random number generators in: pytorch, numpy, python.random 
    pl.seed_everything(seed=456, workers=True)
    # instantiate the model class
    model = MlpModelHeld20Sbs_lrFinder_ManMode(config, root_path, result_dir, pkl_name)
    pkl_name_without_ext = pkl_name.replace('.pkl', '') 
    ckpt_file_nm_1st_part = "MlpModelHeld20Sbs_lrFinder_ManMode_" + pkl_name_without_ext + '-' + str(config["layer_1_size"]) + "-" + str(config["layer_2_size"]) + "-" + str(config["layer_3_size"]) \
    + "-batch" + str(config["batch_size"]) + "-dim" + str(config["reduc_dim"]) + "-lr{:0.4f}-".format(config["lr"])
    ckpt_file_nm_format = ckpt_file_nm_1st_part + "-epoch{epoch:02d}-val_prec{train/val_prec:.3f}"
    
    # define the checkpoint_callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        dirpath=result_dir
                        , filename=ckpt_file_nm_format
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        # , monitor='train/val_loss', mode='min'
                        , monitor='train/val_prec', mode='max'
                        , every_n_epochs = 1, verbose=True)

     # define the early_stop_callback
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        # monitor="train/val_loss"
        monitor="train/val_prec", mode="max"
        , min_delta=0.00, patience=500, verbose=True)  # monitor="val_prec" can be checked

    # # instantiate neptune logger
    # logger = NeptuneLogger(project="sg-neptune/only-seq-prj-v1"
    # , api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzM4ODM3Mi1iZDZjLTQ1MDUtYmYxZi00Mzg3OTA4YTc1NTUifQ=="
    # , name="", prefix='', tags=["training-heldout20-sbs"]
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

    auto_lr_find = False
    if(lr_finder):
        auto_lr_find = True

    # instantiate a trainer
    trainer = pl.Trainer(fast_dev_run=False, max_epochs=num_epochs
                        , deterministic = True
                        , logger = logger
                        # , logger = False
                        , callbacks=[checkpoint_callback, early_stop_callback] # can also use checkpoint_callback, early_stop_callback
                        , accelerator="gpu", devices=1, auto_select_gpus=True  # Deprecated since version v1.7: 'gpus' has been deprecated
                                                #  in v1.7 and will be removed in v2.0. Please use accelerator='gpu' and devices=x instead.
                        , precision = 16
                        # , resume_from_checkpoint = resume_from_checkpoint  # deprecated, use 'ckpt_path' in fit() method instead
                        , auto_lr_find = auto_lr_find
                        , enable_progress_bar = False
                        , enable_model_summary = True)
    if(lr_finder):
        # # to find the best lr
        # trainer.tune(model)
        # print('the best found lr: ' + str(model.lr))
        # return model.lr
        return 0.0003  # keeping fixed for all the datasets

    print('#### before calling trainer.fit(model) method')
    trainer.fit(model, ckpt_path=resume_from_checkpoint)
    print('#### after calling trainer.fit(model) method')
    return None
    # print('#### inside the train_model() method - End')


# consolidate all the best config(s) for all the train datasets in a dataframe and find the 
# minimum lr among them
def consolidate_best_configs(root_path='./'):
    con_best_configs_lst = []
    # train file names list creation
    train_file_name_lst = []
    index_limit = 5
    for first_ind in range(0, index_limit + 1):
        for second_ind in range(first_ind, index_limit + 1):
            indiv_train_fl_nm = 'Train_' + str(first_ind) + '_' + str(second_ind) + '.pkl'
            train_file_name_lst.append(indiv_train_fl_nm)
    print('train_file_name_lst: ' + str(train_file_name_lst))
    # train_file_name_lst = ['Train_0_0.pkl']

    for pkl_name in train_file_name_lst:
        pkl_name_without_ext = pkl_name.replace('.pkl', '') 
        print('\n# ############################# training pkl_name: ' + pkl_name + '\n')
        result_dir = os.path.join(root_path, CHECKPOINTS_DIR, pkl_name_without_ext + '_tune_lr_results')
        best_config_pkl_name_with_path = os.path.join(result_dir, pkl_name_without_ext + '_best_config_dict.pkl')
        # load already saved best config file
        config = joblib.load(best_config_pkl_name_with_path)
        # append this config dictionary in the con_best_configs_lst
        con_best_configs_lst.append(config)
    # end of for loop 
    # convert con_best_configs_lst into a df
    con_best_configs_df = pd.DataFrame(con_best_configs_lst)
    # find the min lr value
    min_lr = con_best_configs_df['lr'].min()
    # save this min lr value as a row of the con_best_configs_df
    con_best_configs_df['min_lr'] = [min_lr] + [''] * (con_best_configs_df.shape[0] - 1)
    # save con_best_configs_df
    con_best_configs_df.to_csv(os.path.join(root_path, CHECKPOINTS_DIR, 'con_best_configs_df.csv'), index=False)

# forming the train and test dataframe from the scaled and dimensionally reduced data-set
def generate_train_test_df_csv(root_path='./'):
    result_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'scaled_reduced_trn_tst_csv')
    # create the result_dir if it does not already exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    reduc_dim = 10
    # train file names list creation
    train_file_name_lst = []
    index_limit = 5
    for first_ind in range(0, index_limit + 1):
        for second_ind in range(first_ind, index_limit + 1):
            indiv_train_fl_nm = 'Train_' + str(first_ind) + '_' + str(second_ind) + '.pkl'
            train_file_name_lst.append(indiv_train_fl_nm)
    print('train_file_name_lst: ' + str(train_file_name_lst))
    for train_pkl_nm in train_file_name_lst:
        print('\n# ############################# training pkl_name: ' + train_pkl_nm + '\n')
        pkl_name_without_ext = train_pkl_nm.replace('.pkl', '') 
        part_preproc_pkl_path = os.path.join(root_path,
        CHECKPOINTS_DIR, pkl_name_without_ext + '_part_preproc_dim_' + str(reduc_dim) + '.pkl')
        print('part_preproc_pkl_path: ' + str(part_preproc_pkl_path))
        # load already saved preproc file
        part_preproc_dict = joblib.load(part_preproc_pkl_path)
        X_train_scaled_reduced = part_preproc_dict['X_train_scaled_reduced']
        y_train_arr = part_preproc_dict['y_train_arr']
        X_test1_scaled_reduced = part_preproc_dict['X_test1_scaled_reduced']
        y_test1_arr = part_preproc_dict['y_test1_arr']
        X_test2_scaled_reduced = part_preproc_dict['X_test2_scaled_reduced']
        y_test2_arr = part_preproc_dict['y_test2_arr']
        # form train and test dataframes and save them as CSV(s)
        print('Forming train and test dataframes and save them as CSV(s)')
        col_nm_lst = ['feat_' + str(idx + 1)  for idx in range(0, reduc_dim)]
        train_df = pd.DataFrame(data=X_train_scaled_reduced, columns=col_nm_lst)
        train_df['target'] = y_train_arr
        test1_df = pd.DataFrame(data=X_test1_scaled_reduced, columns=col_nm_lst)
        test1_df['target'] = y_test1_arr
        test2_df = pd.DataFrame(data=X_test2_scaled_reduced, columns=col_nm_lst)
        test2_df['target'] = y_test2_arr
        train_df_path = os.path.join(result_dir, pkl_name_without_ext + '.csv')
        train_df.to_csv(train_df_path, index=False)
        test1_fl_nm = pkl_name_without_ext.replace('Train', 'Test1')
        test1_df_path = os.path.join(result_dir, test1_fl_nm + '.csv')
        test1_df.to_csv(test1_df_path, index=False)
        test2_fl_nm = pkl_name_without_ext.replace('Train', 'Test2')
        test2_df_path = os.path.join(result_dir, test2_fl_nm + '.csv')
        test2_df.to_csv(test2_df_path, index=False)


def start(root_path='./'):
    config = None
    num_epochs = 1
    # train file names list creation
    train_file_name_lst = []
    index_limit = 5
    for first_ind in range(0, index_limit + 1):
        for second_ind in range(first_ind, index_limit + 1):
            indiv_train_fl_nm = 'Train_' + str(first_ind) + '_' + str(second_ind) + '.pkl'
            train_file_name_lst.append(indiv_train_fl_nm)
    print('train_file_name_lst: ' + str(train_file_name_lst))
    # train_file_name_lst = ['Train_0_0.pkl']
    # TEMP CODE -START
    # train_file_name_lst = [trn_fl_nm for trn_fl_nm in train_file_name_lst if(trn_fl_nm.startswith('Train_3') and trn_fl_nm not in ['Train_3_3.pkl', 'Train_3_4.pkl'])]
    # train_file_name_lst = [trn_fl_nm for trn_fl_nm in train_file_name_lst if(trn_fl_nm.startswith('Train_5'))]
    # TEMP CODE -END

    for pkl_name in train_file_name_lst:
        pkl_name_without_ext = pkl_name.replace('.pkl', '') 
        print('\n# ############################# training pkl_name: ' + pkl_name + '\n')
        result_dir = os.path.join(root_path, CHECKPOINTS_DIR, pkl_name_without_ext + '_tune_lr_results')
        # create the result_dir if it does not already exist
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        best_config_pkl_name_with_path = os.path.join(result_dir, pkl_name_without_ext + '_best_config_dict.pkl')
        # check whether the best config file already exists and saved as a pkl file
        print('checking whether the best config file already exists and saved as a pkl file')
        print('best_config_pkl_name_with_path: ' + str(best_config_pkl_name_with_path))
        if os.path.exists(best_config_pkl_name_with_path):
            # as the the best config file already exists, skipping the best lr value finding part
            print('\n##### As the the best config file already exists, skipping the best lr value finding part...')
            # load already saved best config file
            config = joblib.load(best_config_pkl_name_with_path)
        else:
            # first find the best lr value as no best config file exists
            print('\n##### first find the best lr value as no best config file exists...')
            config = {
            "layer_1_size": 512
            , "layer_2_size": 1024
            , "layer_3_size": 256
            , "batch_size": 64
            # , "reduc_dim": 4484  # 4484  # (1024 + 1024) + (1218 + 1218) using only Li2020
            , "reduc_dim": 3708  # 1854 + 1854
            , "lr": 0.0003  # although finding the best lr value but not using it and keeping a fixed lr value
            }
            # although finding the best lr value but not using it and keeping a fixed lr value
            config["lr"] = train_model(config, root_path, result_dir, pkl_name, num_epochs, lr_finder=True)
            # save the config with the best lr value
            joblib.dump(value=config, filename=best_config_pkl_name_with_path, compress=3)

        # next train using the optimum lr value
        print('next train using the optimum lr value')
        train_model(config, root_path, result_dir, pkl_name, num_epochs, lr_finder=False)
    # end of for loop
    print('\n ##################### END OF THE lr FINDING PROCESS ######################')


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    start(root_path)
    consolidate_best_configs(root_path)
    # generate_train_test_df_csv(root_path)
