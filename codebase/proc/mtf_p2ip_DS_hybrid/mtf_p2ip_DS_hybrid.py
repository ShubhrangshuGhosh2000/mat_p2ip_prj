import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import pandas as pd
from utils import PPIPUtils

root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

orig_dscript_data_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/D_SCRIPT_prj/cross_spec_pred_result')
mtf_p2ip_DS_data_path = os.path.join(root_path, 'dataset/proc_data_DS/DS_hTuned_27epochs_converged')
mtf_p2ip_hybrid_data_path = os.path.join(root_path, 'dataset/proc_data_DS/mtf_p2ip_DS_hybrid')


def create_hybrid_score(spec_type = 'ecoli'):
    print('inside create_hybrid_score() method - Start')
    print('\n########## spec_type: ' + str(spec_type))
    spec_data_path = os.path.join(mtf_p2ip_hybrid_data_path, spec_type)
    PPIPUtils.makeDir(spec_data_path)
    hybrid_pred_prob_1_lst, hybrid_pred_label_lst = [], []
    # read orignal dscript and mtf_p2ip_DS specific prediction probabilities and consolidate them in a single df
    orig_dscript_spec_res_df = pd.read_csv(os.path.join(orig_dscript_data_path, spec_type + '.tsv')
                                            , sep='\t', names = ['prot1', 'prot2', 'dscript_pred_prob_1'])
    mtf_p2ip_DS_spec_res_df = pd.read_csv(os.path.join(mtf_p2ip_DS_data_path, 'mtf_res_origMan_auxTlOtherMan_' + spec_type, 'pred_' + spec_type + '_DS.tsv')
                                           , sep='\t', names = ['mtf_p2ip_DS_pred_prob_1', 'actual_label'])
    combined_df = pd.concat([orig_dscript_spec_res_df, mtf_p2ip_DS_spec_res_df], axis=1)
    
    # iterate over combined_df and apply hybrid strategy
    for index, row in combined_df.iterrows():
        dscript_pred_prob_1= row['dscript_pred_prob_1']
        dscript_pred_prob_0 = 1 - dscript_pred_prob_1
        mtf_p2ip_DS_pred_prob_1 = row['mtf_p2ip_DS_pred_prob_1']
        mtf_p2ip_DS_pred_prob_0 = 1 - mtf_p2ip_DS_pred_prob_1
        # hybrid strategy:
        # Whenever mtf_p2ip_DS is predicting positive and dscript is predicting negative for the same test sample, then the 
        # adjustment is done in such a way so that the prediction which is associated with the higher confidence level (in terms 
        # of the prediction probability) wins. The same is true for the reverse case and for all the other cases, follow mtf_p2ip_DS prediction.
        hybrid_pred_prob_1 = mtf_p2ip_DS_pred_prob_1
        if((mtf_p2ip_DS_pred_prob_1 > 0.5) and (dscript_pred_prob_0 > 0.5)):
            print('\nrow index for applying hybrid strategy: ' + str(index))
            print('mtf_p2ip_DS_pred_prob_1: ' + str(mtf_p2ip_DS_pred_prob_1) + ' : dscript_pred_prob_0: ' + str(dscript_pred_prob_0))
            adj_factor = dscript_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = mtf_p2ip_DS_pred_prob_1 - adj_factor
            print('hybrid_pred_prob_1: ' + str(hybrid_pred_prob_1) + ' : actual_label: ' + str(row['actual_label']))
        elif((mtf_p2ip_DS_pred_prob_0 > 0.5) and (dscript_pred_prob_1 > 0.5)):
            # if mtf_p2ip_DS predicts negative but original dscript predicts positive, then again apply
            # hybrid strategy but in reverse way
            print('\nrow index for applying hybrid strategy: ' + str(index))
            print('mtf_p2ip_DS_pred_prob_1: ' + str(mtf_p2ip_DS_pred_prob_1) + ' : dscript_pred_prob_0: ' + str(dscript_pred_prob_0))
            adj_factor = mtf_p2ip_DS_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = dscript_pred_prob_1 - adj_factor
            print('hybrid_pred_prob_1: ' + str(hybrid_pred_prob_1) + ' : actual_label: ' + str(row['actual_label']))
        
        hybrid_pred_prob_1_lst.append(hybrid_pred_prob_1)
        # now check the hybrid_pred_prob_1 value and set the hybrid prediction label accordingly
        if(hybrid_pred_prob_1 > 0.5):
            hybrid_pred_label_lst.append(1)
        else:
            hybrid_pred_label_lst.append(0)
    # end of for loop: for index, row in combined_df.iterrows():
    combined_df['hybrid_pred_prob_1'] = hybrid_pred_prob_1_lst
    combined_df['hybrid_pred_label'] = hybrid_pred_label_lst
    # save the combined_df
    combined_df.to_csv(os.path.join(spec_data_path, 'combined_df_' + spec_type + '.csv'), index=False) 
    print('\n prediction result processing')
    # compute result metrics, such as AUPR, Precision, Recall, AUROC
    results = PPIPUtils.calcScores_DS(combined_df['actual_label'].to_numpy(), combined_df['hybrid_pred_prob_1'].to_numpy())
    score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            })
    # save score_df as CSV
    score_df.to_csv(os.path.join(spec_data_path, 'score_' + spec_type + '.csv'), index=False) 
    print('inside create_hybrid_score() method - End')


if __name__ == '__main__':
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']
    # spec_type_lst = ['ecoli']
    for spec_type in spec_type_lst:
        create_hybrid_score(spec_type)
