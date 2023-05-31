import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))


import pandas as pd
from utils import PPIPUtils

root_path = os.path.join('/project/root/directory/path/here')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

orig_dscript_data_path = os.path.join('/original_D_SCRIPT_project/cross_species_prediction_result/path/here')
orig_dscript_data_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/D_SCRIPT_prj/cross_spec_pred_result')
mat_p2ip_DS_data_path = os.path.join(root_path, 'dataset/proc_data_DS/DS_hTuned_27epochs_converged')
mat_p2ip_hybrid_data_path = os.path.join(root_path, 'dataset/proc_data_DS/mat_p2ip_DS_hybrid')


def create_hybrid_score(spec_type = 'ecoli'):
    spec_data_path = os.path.join(mat_p2ip_hybrid_data_path, spec_type)
    PPIPUtils.makeDir(spec_data_path)
    hybrid_pred_prob_1_lst, hybrid_pred_label_lst = [], []
    
    orig_dscript_spec_res_df = pd.read_csv(os.path.join(orig_dscript_data_path, spec_type + '.tsv')
                                            , sep='\t', names = ['prot1', 'prot2', 'dscript_pred_prob_1'])
    mat_p2ip_DS_spec_res_df = pd.read_csv(os.path.join(mat_p2ip_DS_data_path, 'mat_res_origMan_auxTlOtherMan_' + spec_type, 'pred_' + spec_type + '_DS.tsv')
                                           , sep='\t', names = ['mat_p2ip_DS_pred_prob_1', 'actual_label'])
    combined_df = pd.concat([orig_dscript_spec_res_df, mat_p2ip_DS_spec_res_df], axis=1)
    for index, row in combined_df.iterrows():
        dscript_pred_prob_1= row['dscript_pred_prob_1']
        dscript_pred_prob_0 = 1 - dscript_pred_prob_1
        mat_p2ip_DS_pred_prob_1 = row['mat_p2ip_DS_pred_prob_1']
        mat_p2ip_DS_pred_prob_0 = 1 - mat_p2ip_DS_pred_prob_1
        
        hybrid_pred_prob_1 = mat_p2ip_DS_pred_prob_1
        if((mat_p2ip_DS_pred_prob_1 > 0.5) and (dscript_pred_prob_0 > 0.5)):
            adj_factor = dscript_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = mat_p2ip_DS_pred_prob_1 - adj_factor
        elif((mat_p2ip_DS_pred_prob_0 > 0.5) and (dscript_pred_prob_1 > 0.5)):
            adj_factor = mat_p2ip_DS_pred_prob_0 - 0.5 
            hybrid_pred_prob_1 = dscript_pred_prob_1 - adj_factor
        
        hybrid_pred_prob_1_lst.append(hybrid_pred_prob_1)
        
        if(hybrid_pred_prob_1 > 0.5):
            hybrid_pred_label_lst.append(1)
        else:
            hybrid_pred_label_lst.append(0)
    
    combined_df['hybrid_pred_prob_1'] = hybrid_pred_prob_1_lst
    combined_df['hybrid_pred_label'] = hybrid_pred_label_lst
    
    combined_df.to_csv(os.path.join(spec_data_path, 'combined_df_' + spec_type + '.csv'), index=False) 
    
    
    results = PPIPUtils.calcScores_DS(combined_df['actual_label'].to_numpy(), combined_df['hybrid_pred_prob_1'].to_numpy())
    score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            })
    score_df.to_csv(os.path.join(spec_data_path, 'score_' + spec_type + '.csv'), index=False) 


if __name__ == '__main__':
    spec_type_lst = ['ecoli', 'fly', 'mouse', 'worm', 'yeast']
    
    for spec_type in spec_type_lst:
        create_hybrid_score(spec_type)
