import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader

path_root = Path(__file__).parents[6]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from codebase.preproc.feat_engg import feat_multiplied_pipr_old
from codebase.utils import test_util
from codebase.utils.custom_2d_feat_dataset import Custom2DfeatDataset
from utils import dl_reproducible_result_util

dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # this is needed for the lambda implementation
CHECKPOINTS_DIR = 'dataset/proc_data/mlp_data/training_data/Random50/feat_multiplied_pipr'
thresholds = [0.03, 1.0]

# PIPR Model for the multiplied(mult) features using Randim50 training sets
class MlpModelRand50MultPipr(pl.LightningModule):

    def __init__(self, config = None, root_path='./', result_dir='./', tsv_name = 'Train.tsv'):
        super(MlpModelRand50MultPipr, self).__init__()
        self.root_path = root_path
        self.result_dir = result_dir
        self.tsv_name = tsv_name

        self.seq_size = config["seq_size"]
        self.reduc_dim = config["reduc_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # define the Residual RCNN layers
        self.__build_res_rcnn_layers()

        # define the MLP layers
        self.__build_mlp_layers()

        # save __init__ arguments to hparams attribute.
        self.save_hyperparameters()


    # define the Residual RCNN layers
    def __build_res_rcnn_layers(self):
        # define 'l' which stands for Conv1D layer as in the PIPR architecture
        # In Conv1D, input is (N,Cin,Lin) and output is (N,Cout,Lout) where
        # N is the batch size,
        # C denotes the number of channels (or the number of features per timestep),
        # L is the length of signal sequence (or the number of timesteps)
        # By default, pytorch data_format = 'channel_first' i.e. (N, C, L) but
        # in Keras, data_format = 'channel_last' i.e. (N, L, C)
        # EXAMPLE: Here in PIPR, each protein is represented with a 2d feature matrix of shape: seq_size(2000) X reduc_dim(18). 
        # So in this case, the number of timesteps or the length of signal sequence in the input, Lin = 2000 and
        # the number of features per timestep or the number of channels in the input, Cin = 18.
        # And for the output, the number of channels, Cout = the number of the 'filters' as in the Keras implmentation of PIPR = self.hidden_dim = 50(say) and
        #  the length of signal sequence or the number of timesteps in the output, Lout = seq_size - kernel_size +1 = 2000 - 3 +1 = 1998
        # INPUT shape: (N,Cin,Lin) = (N, 18(self.reduc_dim), 2000(self.seq_size)); Note: Keep Cin as a placeholder by defining an inline function object.
        # OUTPUT shape: (N,Cout,Lout) = (N, 50(self.hidden_dim), 1998)
        # ## self.l = nn.Conv1d(in_channels=self.reduc_dim, out_channels=self.hidden_dim, kernel_size=3, padding='valid')  # non-lambda implementation
        # In the following lambda implementation, "device" argument needs to be specified as during __init__() method only the
        # lambda object is created and not the nn.Conv1d object and so lightning module cannot transfer this model layer to GPU at
        # the time of model initialization(__init__)
        self.l = lambda cin: nn.Conv1d(in_channels=cin, out_channels=self.hidden_dim, kernel_size=3, padding='valid', device=dev)

        # define 'max_pool' which stands for MaxPooling1D layer as in the PIPR architecture
        # INPUT shape: (N,C,Lin) = (N, 50(self.hidden_dim), 1998);
        # OUTPUT shape: (N,C,Lout) = (N, 50(self.hidden_dim), 666) where Lout = floor((1998 - 3)/3 + 1) = 666
        self.max_pool = nn.MaxPool1d(kernel_size=3)

        # define 'r' which stands for Bidirectional CuDNNGRU layer as in the PIPR architecture
        # In GRU, when batch_first=True, input is (N,L,Hin​) and output are (N,L,D∗Hout​); (D∗num_layers,N,Hout​) where
        # N is the batch size,
        # L is the length of signal sequence (or the number of the timesteps)
        # Hin denotes the input_size (or the number of features per timestep)
        # Hout denotes the hidden_size (or the number of features in a hidden state)
        # and D is 2 if bidirectional=True otherwise 1
        # INPUT shape: (N,L,Hin​) = (N, 666, 50(self.hidden_dim)); Note: self.max_pool layer output needs to be transposed properly to feed that as input. 
        # OUTPUTs shape: (N,L,D∗Hout​) = (N, 666, 2*50(self.hidden_dim)); (D∗num_layers,N,Hout​) = (2*1, N, 50((self.hidden_dim)))
        self.r = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True)

        # define 'cat' which stands for concatenate as in the PIPR architecture
        # INPUTs shape: 2 inputs - the first input from nn.GRU layer output i.e. (N,L,D∗Hout​) = (N, 666, 2*50(self.hidden_dim)) and the second
        # input is from nn.MaxPool1d layer output i.e. (N,C,Lout) = (N, 50(self.hidden_dim), 666).
        # Note: The 1st input needs to be transposed as (N,D∗Hout​,L) = (N, 2*50(self.hidden_dim), 666) and the concatenation would be carried over the dim=1
        # OUTPUT shape: (N,Cout,L) = (N, 150, 666) where Cout = 2*50(self.hidden_dim) + 50(self.hidden_dim) = 3 * 50(self.hidden_dim) = 150
        self.cat = torch.cat

        # define 'glob_avg_pool' which stands for GlobalAveragePooling1D layer as in the PIPR architecture
        # INPUT shape: (N,C,Lin) as already defined above in 'self.l'
        # OUTPUT shape: (N,C,Lout) where Lout=output_size=1 as per the definition of GlobalAveragePooling1D in Keras. So
        # the output shape will be (N,C,1). 
        # Note: The last dimension should then need to be squeezed to make the final output shape as (N,C) which
        # is as per with Keras implemented PIPR architecture
        self.glob_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)


    # define the MLP layers
    def __build_mlp_layers(self):
        # setting up the Sequential for MLP layer
        # self.mlp = nn.Sequential(  # non-lambda implementation
        self.mlp = lambda inp_feat: nn.Sequential(  # lambda implementation; "device" argument needs to be specified in every nn.Linear layer
                                        nn.Linear(in_features=inp_feat, out_features=100, device=dev),
                                        # ## nn.Linear(in_features=self.hidden_dim, out_features=100),  # non-lambda implementation
                                        nn.LeakyReLU(negative_slope=0.3, inplace=True),

                                        nn.Linear(in_features=100, out_features=int((self.hidden_dim+7)/2), device=dev),
                                        nn.LeakyReLU(negative_slope=0.3, inplace=True),
                                        nn.Linear(in_features=int((self.hidden_dim+7)/2), out_features=2, device=dev),
                                        nn.Softmax(dim=1)
                                    )  # end of nn.Sequential


    # this method will be called from the forward() method once for each of the 2 proteins batch
    def __forward_once(self, x):
        # print('#### inside the forward_once() method')
        batch_size, two_d_feat_row_size, two_d_feat_col_size = x.size()
        # change shape from (b, 2000, 18) to (b, 18, 2000) as by default, pytorch data_format = 'channel_first' i.e. (N, C, L) 
        x = x.view(batch_size, two_d_feat_col_size, two_d_feat_row_size)
        # print("After changing to 'channel_first' format x.shape : " + str(x.shape))

        # # simulate the following Keras implementation from the PIPR paper:
        # # s1=MaxPooling1D(3)(l1(seq_input1))
        # # s1=concatenate([r1(s1), s1])  #1
        # # s1=MaxPooling1D(3)(l2(s1))
        # # s1=concatenate([r2(s1), s1])  #2
        # # s1=MaxPooling1D(3)(l3(s1))
        # # s1=concatenate([r3(s1), s1])  #3
        # # s1=MaxPooling1D(3)(l4(s1))
        # # s1=concatenate([r4(s1), s1])  #4
        # # s1=MaxPooling1D(3)(l5(s1))
        # # s1=concatenate([r5(s1), s1])  #5
        # # s1=l6(s1)
        # # s1=GlobalAveragePooling1D()(s1)

        s = self.max_pool(self.l(self.reduc_dim)(x))
        r1_s, __ = self.r(s.transpose(1,2))
        s = self.cat((r1_s.transpose(1,2), s), dim=1)  #1

        s = self.max_pool(self.l(s.shape[1])(s))
        r1_s, __ = self.r(s.transpose(1,2))  # check shape of s => (256, 50, 221)
        s = self.cat((r1_s.transpose(1,2), s), dim=1)  #2

        s = self.max_pool(self.l(s.shape[1])(s))
        r1_s, __ = self.r(s.transpose(1,2))
        s = self.cat((r1_s.transpose(1,2), s), dim=1)  #3

        s = self.max_pool(self.l(s.shape[1])(s))
        r1_s, __ = self.r(s.transpose(1,2))
        s = self.cat((r1_s.transpose(1,2), s), dim=1)  #4

        s = self.max_pool(self.l(s.shape[1])(s))
        r1_s, __ = self.r(s.transpose(1,2))
        s = self.cat((r1_s.transpose(1,2), s), dim=1)  #5
        s = self.l(s.shape[1])(s)
        s = self.glob_avg_pool(s)
        s = s.squeeze(dim=-1)
        return s


    def forward(self, x1, x2,):
        # in lightning, forward defines the prediction/inference actions
        # print('#### inside the forward() method')
        # print('x1.size(): ' + str(x1.size()))
        batch_size, two_d_feat_row_size, two_d_feat_col_size = x1.size()
        # (b, 2000, 18) -> (b, 2000, 18)
        x1 = x1.view(batch_size, two_d_feat_row_size, two_d_feat_col_size)
        x2 = x2.view(batch_size, two_d_feat_row_size, two_d_feat_col_size)
        # call the forward_once() method for each of the 2 proteins batch
        s1 = self.__forward_once(x1)
        s2 = self.__forward_once(x2)
        # multiply element-wise both the embeddings s1, s2
        merge_text = torch.mul(s1, s2)
        # concatenate side-by-side both the embeddings s1, s2
        # #### merge_text = torch.cat((s1, s2), axis=1)
        # print('merge_text.shape: ' + str(merge_text.shape))
        # call MLP layers
        main_output = self.mlp(merge_text.shape[1])(merge_text)
        return main_output


    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)


    # normal accuracy calculation
    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)


    # special accuracy calculation
    def accuracy_spl(self, logits, labels):
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
        accuracy = score_dict['ACC']
        return torch.tensor(accuracy)


    def training_step(self, train_batch, batch_idx):
        # REQUIRED
        # training_step defined the train loop.
        # It is independent of forward
        # print('#### inside the training_step() method')
        x1, x2, y = train_batch
        logits = self.forward(x1, x2)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log("train/train_loss", loss)
        self.log("train/train_accuracy", accuracy)
        return {'loss': loss, 'accuracy': accuracy}


    def validation_step(self, val_batch, batch_idx):
        # OPTIONAL
        # print('#### inside the validation_step() method')
        x1, x2, y = val_batch
        logits = self.forward(x1, x2)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": accuracy}


    def validation_epoch_end(self, outputs):
        # OPTIONAL
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 

        print('#### inside the validation_epoch_end() method')
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("train/val_loss", avg_loss)  # ## this is important for model checkpointing
        self.log("train/val_accuracy", avg_acc)
        print('train/val_accuracy: ' + str(avg_acc.numpy()))
        return {'val_loss': avg_loss, "val_accuracy": avg_acc}


    def predict_step(self, pred_batch, batch_idx, dataloader_idx=0):
        # OPTIONAL
        # By default, the predict_step() method runs the forward() method.
        # In order to customize this behaviour, simply override the predict_step() method.
        # In the case where you want to scale your inference, you should be using predict_step()
        x1, x2, __ = pred_batch
        x1, x2 = x1.to(dev), x2.to(dev)
        # this calls forward
        logits = self.forward(x1, x2)
        return logits


    def configure_optimizers(self):
        # REQUIRED
        # print('#### inside the configure_optimizers() method')
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True, eps=1e-6)
        return optimizer


    def prepare_data(self):
        # Even in multi-GPU training, this method is called only from a single GPU. 
        # So this method ideal for download, stratification etc.

        print('#### inside the prepare_data() method -Start')
        # call the data preparation method for the training tsv file 
        two_d_feat_arr_1_trn, two_d_feat_arr_2_trn, label_arr_1d_trn = feat_multiplied_pipr_old.data_prep_feat_multiplied_pipr(root_path, 
        dataset_type='Random50', tsv_file_nm=self.tsv_name)
        two_d_feat_arr_1_trn, two_d_feat_arr_2_trn, label_arr_1d_trn = two_d_feat_arr_1_trn.astype(np.float32),\
            two_d_feat_arr_2_trn.astype(np.float32), label_arr_1d_trn.astype(int)
        # ######################### TEMP CODE -START #########################
        val_tsv_name = self.tsv_name.replace('Train', 'Test')
        two_d_feat_arr_1_val, two_d_feat_arr_2_val, label_arr_1d_val = feat_multiplied_pipr_old.data_prep_feat_multiplied_pipr(root_path, 
        dataset_type='Random50', tsv_file_nm=val_tsv_name)
        two_d_feat_arr_1_val, two_d_feat_arr_2_val, label_arr_1d_val = two_d_feat_arr_1_val.astype(np.float32),\
            two_d_feat_arr_2_val.astype(np.float32), label_arr_1d_val.astype(int)
        # ######################### TEMP CODE -END #########################
        # create the custom torch dataset to be used by the torch dataloader
        print('creating the custom torch dataset to be used by the torch dataloader')
        self.train_data = Custom2DfeatDataset(two_d_feat_arr_1_trn, two_d_feat_arr_2_trn, label_arr_1d_trn)
        self.val_data = Custom2DfeatDataset(two_d_feat_arr_1_val, two_d_feat_arr_2_val, label_arr_1d_val)
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

def train_model(config = None, root_path='./', result_dir='./', tsv_name = 'train.tsv', num_epochs = 10):
    print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random number generators in: pytorch, numpy, python.random 
    pl.seed_everything(seed=456, workers=True)
    # instantiate the model class
    model = MlpModelRand50MultPipr(config, root_path, result_dir, tsv_name)
    tsv_name_without_ext = tsv_name.replace('.tsv', '') 
    ckpt_file_nm_1st_part = "MlpModelRand50MultPipr_" + tsv_name_without_ext
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
        , min_delta=0.00, patience=500, verbose=True)  # monitor="val_acc" can be checked

    # # instantiate neptune logger
    # logger = NeptuneLogger(project="sg-neptune/only-seq-prj-v1"
    # , api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMzM4ODM3Mi1iZDZjLTQ1MDUtYmYxZi00Mzg3OTA4YTc1NTUifQ=="
    # , name="", prefix='', tags=["training-random50-multPipr"]
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
                        , callbacks=[checkpoint_callback, early_stop_callback]  # can also use checkpoint_callback, early_stop_callback
                        , accelerator="gpu", devices=1, auto_select_gpus=True  # Deprecated since version v1.7: 'gpus' has been deprecated
                                                #  in v1.7 and will be removed in v2.0. Please use accelerator='gpu' and devices=x instead.
                        , precision = 16
                        # , resume_from_checkpoint = resume_from_checkpoint  # deprecated, use 'ckpt_path' in fit() method instead
                        , auto_lr_find = False
                        , enable_progress_bar = False
                        , enable_model_summary = True)
    print('#### before calling trainer.fit(model) method')
    trainer.fit(model, ckpt_path=resume_from_checkpoint)
    print('#### after calling trainer.fit(model) method')
    print('#### inside the train_model() method - End')


def start(root_path='./'):
    config = None
    num_epochs = 50

    no_of_train_files = 5
    # for ind in range(0, no_of_train_files):
    for ind in range(0, 1):
        tsv_name = 'Train_' + str(ind) + '.tsv'
        tsv_name_without_ext = tsv_name.replace('.tsv', '') 
        print('\n# ############################# training tsv_name: ' + tsv_name + '\n')
        result_dir = os.path.join(root_path, CHECKPOINTS_DIR, tsv_name_without_ext + '_results')
        # create the result_dir if it does not already exist
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        config = {
        "seq_size": 2000  # it is always 2000
        , "reduc_dim": 18
        , "hidden_dim": 50
        , "batch_size": 256
        , "lr": 0.001
        }
        train_model(config, root_path, result_dir, tsv_name, num_epochs)
    # end of for loop
    print('\n ##################### END OF THE TRAINING PROCESS ######################')


if __name__ == '__main__':
    # root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    start(root_path)
