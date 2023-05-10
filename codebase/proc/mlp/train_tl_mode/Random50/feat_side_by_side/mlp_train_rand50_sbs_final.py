import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytorch_lightning as pl
import torch
import umap
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from sklearn import model_selection, preprocessing
from torch.nn import functional as F
from torch.utils.data import DataLoader
import glob

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.utils.custom_dataset import CustomDataset
from utils import dl_reproducible_result_util

CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side'

# Multilayer Perceptron (MLP) Model for the Side-By-Side(sbs) features using Randim50 training sets 
class MlpModelRand50SbsFinal(pl.LightningModule):

    def __init__(self, root_path='./', pkl_name = 'train.pkl', config=None):
        super(MlpModelRand50SbsFinal, self).__init__()
        self.root_path = root_path
        self.pkl_name = pkl_name
        # set the hyper-params
        self.layer_1_size = config['layer_1_size']
        self.layer_2_size = config['layer_2_size']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reduc_dim = config['reduc_dim']

        # Defining the model architecture
        # for side-by-side (sbs), feature vector length is 2048 (1024 for 
        # each interacting protein) 

        self.layer_1 = torch.nn.Linear(self.reduc_dim, self.layer_1_size)
        self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
        self.layer_3 = torch.nn.Linear(self.layer_2_size, 2)

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
        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):
        # REQUIRED
        # training_step defined the train loop.
        # It is independent of forward
        # print('#### inside the training_step() method')
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        # self.log("train/train_loss", loss)
        # self.log("train/train_accuracy", accuracy)
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, val_batch, batch_idx):
        # OPTIONAL
        print('#### inside the validation_step() method')
        # x, y = val_batch
        # logits = self.forward(x)
        # loss = F.nll_loss(logits, y)
        # self.log('val/loss', loss)
        # return {'val_loss': loss}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        # called at the end of the training epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 

        print('\n#### inside the training_epoch_end() method')
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("train/train_loss", avg_loss)  # ## this is important for model checkpointing
        self.log("train/train_accuracy", avg_acc)
        # return {'train_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        print('#### inside the configure_optimizers() method')
        return [torch.optim.Adam(self.parameters(), lr=self.lr)]

    def prepare_data(self):
        # Even in multi-GPU training, this method is called only from a single GPU. 
        # So this method ideal for download, stratification etc.

        print('#### inside the prepare_data() method -Start')
        X_train_scaled_reduced = y_train_arr = None
        # check whether the part of preprocessing (upto dim-reduction) is already done and saved as a pkl file
        print('checking whether the part of preprocessing (upto dim-reduction) is already done and saved as a pkl file')
        pkl_name_without_ext = self.pkl_name.replace('.pkl', '') 
        part_preproc_pkl_path = os.path.join(self.root_path,
        'dataset/proc_data/mlp_data/training_data/Random50/feat_side_by_side', pkl_name_without_ext + '_part_preproc_final.pkl')
        print('part_preproc_pkl_path: ' + str(part_preproc_pkl_path))
        if os.path.exists(part_preproc_pkl_path):
            # as the partial preproc file already exists, skipping upto dim-reduction step
            print('\n##### As the partial preproc file already exists, skipping upto dim-reduction step...')
            # load already saved preproc file
            part_preproc_dict = joblib.load(part_preproc_pkl_path)
            X_train_scaled_reduced = part_preproc_dict['X_train_scaled_reduced']
            y_train_arr = part_preproc_dict['y_train_arr']
            self.scaler = part_preproc_dict['scaler']
        else:
            # start data preparation from the beginning as no partial preproc file exists
            print('\n##### starting data preparation from the beginning as no partial preproc file exists...')
            training_data_path = os.path.join(self.root_path,
            'dataset/preproc_data/human_2021/Random50/feat_side_by_side', self.pkl_name)
            print('training_data_path: ' + str(training_data_path))
            # load the pkl file
            print('loading the pkl file')
            train_lst = joblib.load(training_data_path)
            # train_lst is a lst of 1d arrays; now convert it into a 2d array
            train_arr_2d = np.vstack(train_lst)
            # next perform column filtering and column rearranging so that the feature columns come first and then
            # the target column (in train_arr_2d, the target column is in the 2th column index and features are started
            # from 3th column index onwards)
            train_arr = train_arr_2d[:, list(range(3, train_arr_2d.shape[1])) + [2]]
            X_train_arr = train_arr[:, range(0, train_arr.shape[1] -1)]  # excluding the target column
            y_train_arr = train_arr[:, -1]  # the last column i.e. target column
            # z-normalize X_train_arr feature(column) wise
            print('z-normalizing X_train_arr feature(column) wise')
            scaler = preprocessing.StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_arr)
            # save the scaler for the test data normalization at a later stage
            self.scaler = scaler
            # perform the dimensionality reduction on the training data
            print('performing the dimensionality reduction on the training data')
            print('reduc_dim = ' + str(self.reduc_dim))
            # initiate the umap
            dim_reducer = umap.UMAP(n_components=self.reduc_dim, low_memory=False, n_epochs=400, densmap=False
                                    , n_neighbors=30, random_state=456, verbose=True)
            # apply fit on the X_train_scaled and y_train_arr
            dim_reducer.fit(X_train_scaled, y_train_arr)
            # apply transform on the X_train_scaled
            X_train_scaled_reduced = dim_reducer.transform(X_train_scaled)
            # save the result upto the dim-reduction step as the partial preproc file so that it can be reused 
            print('saving the result upto the dim-reduction step as the partial preproc file so that it can be reused')
            joblib.dump(value={'X_train_scaled_reduced': X_train_scaled_reduced, 'y_train_arr': y_train_arr
                               , 'scaler': scaler
                               # , 'dim_reducer': dim_reducer  # DON'T save dim_reducer as it will make the process vary slow as
                               # it would save kNN model with all the training data. Instead, recreate it during the testing time 
                               # using the same TRAINING data with the same random_state and then apply the same caching strategy
                               # for the dimensionally reduced test data like training data.
                               }
                        , filename=part_preproc_pkl_path
                        , compress=3)

        # do NOT splits the data into training and validation sets for the final model preparation
        print('NOT splitting the data into training and validation sets for the final model preparation')
        # X_train, X_val, y_train, y_val = model_selection.train_test_split(
        #     X_train_scaled_reduced, y_train_arr, test_size=0.2, random_state=456)
        X_train = X_train_scaled_reduced
        y_train = y_train_arr

        # tranform the 2d numpy arrays to the torch tensors
        print('transforming the 2d numpy arrays to the torch tensors')
        X_train = X_train.astype(np.float32)
        X_train = torch.from_numpy(X_train)
        # transform the 1d numpy array of floats to the array of int as the target label is integer
        # no need for np.round() here as the floats are either 0.0 or 1.0
        y_train = y_train.astype(int)
        # create the custom torch dataset to be used by the torch dataloader
        print('creating the custom torch dataset to be used by the torch dataloader')
        self.train_data = CustomDataset(X_train, y_train)
        print('#### inside the prepare_data() method -End')

    def train_dataloader(self):
        print('#### inside the train_dataloader() method')
        return DataLoader(self.train_data, batch_size=self.batch_size
        , num_workers=os.cpu_count(), pin_memory= True if(torch.cuda.is_available()) else False
        , shuffle=True)

    def val_dataloader(self):
        print('#### inside the val_dataloader() method')
        # return DataLoader(self.val_data, batch_size=self.batch_size
        # , num_workers=os.cpu_count(), pin_memory= True if(torch.cuda.is_available()) else False)

def train_model(root_path='./', pkl_name = 'train.pkl', num_epochs=4, config=None):
    print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random number generators in: pytorch, numpy, python.random 
    pl.seed_everything(seed=456, workers=True)

    # instantiate the model class
    model = MlpModelRand50SbsFinal(root_path, pkl_name, config)

    # define the checkpoint_callback
    pkl_name_without_ext = pkl_name.replace('.pkl', '') 
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(root_path, CHECKPOINTS_DIR, 'final_model')
                        , filename="Final-MlpModelRand50Sbs-" + str(pkl_name_without_ext) + "-epoch{epoch:02d}-train_loss{train/train_loss:.2f}-acc{train/train_accuracy:.2f}"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        # , monitor='train/train_loss', mode='min'
                        , monitor='train/train_accuracy', mode='max'
                        , every_n_epochs = 1, verbose=True)

     # define the early_stop_callback
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        # monitor="train/train_loss", mode='min', stopping_threshold=0.0, min_delta=0.0, patience=15, check_on_train_epoch_end=True, verbose=True)  # monitor="val_acc" can be checked
        monitor="train/train_accuracy", mode='max', stopping_threshold=1.0, min_delta=0.0, patience=15, check_on_train_epoch_end=True, verbose=True)  # monitor="val_acc" can be checked

    # instantiate neptune logger
    logger = NeptuneLogger(project="sg-neptune/only-seq-prj-v1"
    , api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzM4ODM3Mi1iZDZjLTQ1MDUtYmYxZi00Mzg3OTA4YTc1NTUifQ=="
    , name="training-random50-sbs-final", prefix='training-random50-sbs-final', tags=["training-random50-sbs-final"]
    )

    # instantiate tensorboard logger
    # tb_dir = os.path.join(root_path, CHECKPOINTS_DIR, 'final_tb_logs', pkl_name_without_ext + '_final_model')
    # logger = TensorBoardLogger(tb_dir, name="training-random50-sbs-final", prefix='training-random50-sbs-final', log_graph=True)

    # instantiate a trainer
    trainer = pl.Trainer(fast_dev_run=False, max_epochs=num_epochs
                        , deterministic = True
                        , logger = logger
                        # , logger = False
                        , callbacks=[checkpoint_callback, early_stop_callback]
                        , gpus = -1 if(torch.cuda.is_available()) else 0
                        , precision =16
                        , enable_progress_bar = False
                        , enable_model_summary = True)
    print('#### before calling trainer.fit(model) method')
    trainer.fit(model)
    print('#### after calling trainer.fit(model) method')
    print('#### inside the train_model() method - End')


def start(root_path='./', pkl_name = 'train.pkl' ):
    num_epochs = 100
    print('\n# ############################# Final training pkl_name: ' + pkl_name + '\n')

    # load the best_result_dict saved during the hyper-param tuning and populate the best-fit hyper params
    print('loading the best_result_dict saved during the hyper-param tuning and populating the best-fit hyper params')
    pkl_name_without_ext = pkl_name.replace('.pkl', '') 
    hparam_result_dir = os.path.join(root_path, CHECKPOINTS_DIR, pkl_name_without_ext + '_tune_results')
    best_result_dict = joblib.load(os.path.join(hparam_result_dir, 'best_result_dict.pkl'))
    config = best_result_dict['config']
    print('best-fit hyper-prams: ' + str(config))

    # ############ USE ONLY IN THE SPECIAL CASE, WHEN THE BEST H-PARAM COMBINATION IS NOT SAVED IN THE REGULAR FASHION.
    # OTHERWISE MUST BE COMMENTED OUT.
    # config = {'lr' : 0.00046545178487942656, 'batch_size' : 256, 'reduc_dim': 10, 'layer_1_size': 256, 'layer_2_size' : 256, 'layer_3_size' : 512}

    print('removing pre-existing final checkpoints (if any)...')
    # Get a list of all the file paths that ends with .ckpt in specified directory
    ckpt_fileList = glob.glob(os.path.join(root_path, CHECKPOINTS_DIR, 'final_model', "Final-MlpModelRand50Sbs-" + str(pkl_name_without_ext) + '*.ckpt'), recursive=False)
    # Iterate over the list of filepaths & remove each file.
    for ckptFilePath in ckpt_fileList:
        try:
            os.remove(ckptFilePath)
        except:
            print("Error while deleting file : ", ckptFilePath)
    # next call train_model() method
    train_model(root_path, pkl_name, num_epochs, config)

if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    no_of_train_files = 5
    # for ind in range(0, no_of_train_files):
    for ind in range(0, 1):
        pkl_name = 'Train_' + str(ind) + '.pkl'
        start(root_path, pkl_name)
