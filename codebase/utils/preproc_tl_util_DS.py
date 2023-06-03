import gc
import glob
import os
import re

import joblib
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

root_path = os.path.join('/project/root/directory/path/here')


# load ProtTrans tl-model for the given type of model
def load_protTrans_tl_model(protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50'):
    print("\n ################## loading protTrans tl-model ##################")
    # 1. Load necessry libraries including huggingface transformers
    # from transformers import T5EncoderModel, T5Tokenizer
    # import gc

    # 2. Load the vocabulary and ProtTrans Model
    protTrans_model_name_with_path = os.path.join(protTrans_model_path, protTrans_model_name)
    tokenizer = T5Tokenizer.from_pretrained(protTrans_model_name_with_path, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(protTrans_model_name_with_path)
    gc.collect()
    # return the loaded model and tokenizer
    print('returning the loaded model and tokenizer')
    return (model, tokenizer) 

# use ProtTrans model(s) to extract the features for a given list of sequences 
def extract_feat_from_protTrans(seq_lst, protTrans_model_path='./', protTrans_model_name = 'prot_t5_xl_uniref50', spec_type = 'human'):
    # first load the vocabulary and ProtTrans Model which will be used subsequently
    model, tokenizer = load_protTrans_tl_model(protTrans_model_path, protTrans_model_name)
    # next extract the features using the loaded model
    features = extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, spec_type, device='gpu')
    # return the feature list
    print('returning the consolidated feature list')
    return features

# use preloaded ProtTrans model(s) to extract the features for a given list of sequences 
def extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, spec_type, device='gpu'):
    print('\n########## spec_type: ' + str(spec_type))
    # first create a list of compact sequences without the whitespace in between the characters
    compact_seq_lst = [seq.replace(" ", "") for seq in seq_lst]
    # then create a sequence list where the characters in each sequence is separated by a single whitespace
    seq_lst_with_space = [' '.join(seq) for seq in compact_seq_lst]
    # print("seq_lst_with_space:\n " + str(seq_lst_with_space))
    
    # invoke protTrans model for the feature extraction
    print("\n ################## invoking protTrans model for the feature extraction ##################")

    # 3. Load the model into the GPU if avilabile and switch to inference mode
    if(device == 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()

    # 4. Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
    # seq_lst = ["A E T C Z A O","S K T Z P"]
    seq_lst_with_space = [re.sub(r"[UZOB]", "X", sequence) for sequence in seq_lst_with_space]

    # partition size indicates the number of the sequences which will be present in each partition (except might be for the last partition)
    part_size = 1
    print("partitioning the original sequence list into a number of sublists where each sublist (except the last) contains " + str(part_size) + " sequences...")
    part_lst = [seq_lst_with_space[i: i + part_size] for i in range(0, len(seq_lst_with_space), part_size)]
    tot_no_of_parts = len(part_lst)
    print('original sequence list length = ' + str(len(seq_lst_with_space)))
    print('total number of partitions (each of size ' + str(part_size) + ') = ' + str(tot_no_of_parts))

    temp_feat_lst = []  # for the intermediate saving purpose
    temp_result_dir = os.path.join(root_path, 'temp_result_' + spec_type)  # for the iteration-wise intermediate pkl file saving purpose
    temp_per_prot_emb_result_dir = os.path.join(root_path, 'temp_per_prot_emb_result_' + spec_type)  # for the iteration-wise intermediate per protein embedding list saving purpose

    # find and set the latest_pkl_file_index as present in the temp_result_dir
    # this latest_pkl_file_index will help in skipping iteration, if it is already done (see below)
    latest_pkl_file_index = -1
    # fetching all the pkl files with name starting with 'feat_lst_'
    all_pkl_fl_nm_lst = glob.glob(os.path.join(temp_result_dir, 'feat_lst_*.pkl'))
    if(len(all_pkl_fl_nm_lst) > 0):  # temp_result_dir is not empty
        # finding the latest pkl file name
        pkl_file_ind_lst = [int(indiv_fl_nm.replace(os.path.join(temp_result_dir, 'feat_lst_'), '').replace('.pkl', '')) for indiv_fl_nm in all_pkl_fl_nm_lst]
        # latest_pkl_fl_nm = max(all_pkl_fl_nm_lst, key=os.path.getctime)
        # extracting the max iteration index
        latest_pkl_file_index = max(pkl_file_ind_lst)
    print('##### latest_pkl_file_index: ' + str(latest_pkl_file_index))

    cuda_error = False
    # for itr in range(0, tot_no_of_parts):
    for itr in range(latest_pkl_file_index + 1, tot_no_of_parts):
        print('ítr: ' + str(itr))
        indiv_part_lst = part_lst[itr]
        # 5. Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(indiv_part_lst, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 6. Extracting sequences' features and load it into the CPU if needed
        try:
            with torch.no_grad():
                embedding = model(input_ids=input_ids,attention_mask=attention_mask)
            embedding = embedding.last_hidden_state.cpu().numpy()
        except Exception as ex:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@ inside exception@@@@@@@@@@@@@@@@@@')
            err_message = str(ex) + ".....For details, please enable 'verbose' argument as True (if not already enabled) and see the log."
            print("Error message :", err_message, sep="")
            cuda_error = True  # 'cpu' should be used for the next iteration
            break  # jump out of the for loop so that 'cpu' can be used for the next iteration

        # 7. Remove padding (<pad>) and special tokens (</s>) that is added by ProtTrans model
        for seq_num in range(len(embedding)):
            # print('seq_num = ' + str(seq_num) + ' out of ' + str(len(embedding)))
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            temp_feat_lst.append(seq_emd)
        print('######### completed ' + str(itr) + '-th iteration out of ' + str(tot_no_of_parts-1))
        # after every 100th iteration or at the last iteration, save the temp_feat_lst and reset it
        if((itr % 1 == 0) or (itr in [(tot_no_of_parts - 1)])):
            print("\n Saving intermediate result to a .pkl file...")
            filename = os.path.join(temp_result_dir, 'feat_lst_' + str(itr) + '.pkl')
            joblib.dump(value=temp_feat_lst, filename=filename, compress=0)
            # reset the temp_feat_lst
            temp_feat_lst = []
        # cpu should be used only in that iteration for which gpu memory is not sufficient and
        # after that again gpu will be used
        if(device == 'cpu'):
            # call this method again but indicate the device should be gpu
            return extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='gpu')
    # end of for itr in range(0, tot_no_of_parts)

    # check for the cuda-error
    if(cuda_error):
        # call this method again but indicate device should be cpu
        return extract_feat_from_preloaded_protTrans(seq_lst, model, tokenizer, device='cpu')

    # add all the intermediate .pkl files in order to create the final feature list
    print('adding all the intermediate .pkl files in order to create the final feature list')
    features = []  # for the pooled 1d features per protein
    loop_start_index = 0  # #### this can be hardcoded in case of the for loop restart 
    for itr in range(loop_start_index, tot_no_of_parts):
        if((itr % 1 == 0) or (itr in [(tot_no_of_parts - 1)])):
            # perform pooling column-wise to create fixed-size protein level embeddings from the residue level embeddings per protein
            # print('performing pooling column-wise to create fixed-size protein level embeddings from the residue level embeddings per protein')
            if((itr > 141000) or (itr % 200== 0)) : print('performing pooling column-wise :: itr = ' + str(itr))
            # print('performing pooling column-wise :: itr = ' + str(itr))
            temp_feat_lst = joblib.load(os.path.join(temp_result_dir, 'feat_lst_' + str(itr) + '.pkl'))
            for prot_residue_embed in temp_feat_lst:  # each element of temp_feat_lst is a 2d array containing residue level embeddings per protein
                # apply pooling column-wise and store the resulting 1d array of fixed size (1024) in the features
                features.append(np.apply_along_axis(np.median, axis=0, arr=prot_residue_embed))  # can apply np.mean/max/min etc. in place np.median
            # end of for loop
            # after every 20k iterations save the intermediate 'features' list and 'features_2d' list in the temp_per_prot_emb_result_dir so that
            # in case of the 'for loop' restart, the iterations which are already done can be skipped.  
            if(itr % 20000 == 0): 
                temp_feat_lst_file_nm = os.path.join(temp_per_prot_emb_result_dir, 'features_' + str(loop_start_index) + '_' + str(itr) + '.pkl')
                joblib.dump(value=features, filename=temp_feat_lst_file_nm, compress=0)
    # end of for loop
    # save the final feature list and features_2d list  as a pkl file so that it can be reused, if needed
    joblib.dump(value=features, filename=os.path.join(temp_per_prot_emb_result_dir, 'features.pkl'), compress=0)
    # return the feature list
    print("\n ################## returning the consolidated 1d feature list")
    return features


if __name__ == '__main__':
    # print('in main')
    pass
