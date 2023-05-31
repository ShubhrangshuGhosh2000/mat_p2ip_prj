import sys, os
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


def calc_attribution_score(root_path = './'):
    print('inside calc_attribution_score() method -start')
    spec_type_lst = ['mouse', 'fly', 'worm', 'yeast', 'ecoli']  
    tl_1d_overall_attrbn_lst, tl_1d_corr_pred_attrbn_lst, tl_1d_incorr_pred_attrbn_lst = [], [], []
    man_2d_overall_attrbn_lst, man_2d_corr_pred_attrbn_lst, man_2d_incorr_pred_attrbn_lst = [], [], []
    man_1d_overall_attrbn_lst, man_1d_corr_pred_attrbn_lst, man_1d_incorr_pred_attrbn_lst = [], [], []
    man_overall_attrbn_lst, man_corr_pred_attrbn_lst, man_incorr_pred_attrbn_lst = [], [], []

    for spec_type in spec_type_lst:
        print('\n########## spec_type: ' + str(spec_type))
        spec_attrbn_file_path = os.path.join(root_path, 'dataset/proc_data_DS/mat_res_origMan_auxTlOtherMan_' + spec_type,
                                            'attribution_' + spec_type + '.csv')
        spec_attrbn_df = pd.read_csv(spec_attrbn_file_path)

        
        tl_1d_overall_attrbn = spec_attrbn_df['tl_1d_feat_attrbn'].mean()
        tl_1d_overall_attrbn_lst.append(tl_1d_overall_attrbn)
        man_2d_overall_attrbn = spec_attrbn_df['man_2d_feat_attrbn'].mean()
        man_2d_overall_attrbn_lst.append(man_2d_overall_attrbn)
        man_1d_overall_attrbn = spec_attrbn_df['man_1d_feat_attrbn'].mean()
        man_1d_overall_attrbn_lst.append(man_1d_overall_attrbn)
        man_overall_attrbn_lst.append(man_2d_overall_attrbn + man_1d_overall_attrbn)

        
        corr_pred_df = spec_attrbn_df[spec_attrbn_df['pred_label'] == spec_attrbn_df['actual_label']]
        corr_pred_df =  corr_pred_df.reset_index(drop=True)
        tl_1d_corr_pred_attrbn = corr_pred_df['tl_1d_feat_attrbn'].mean()
        tl_1d_corr_pred_attrbn_lst.append(tl_1d_corr_pred_attrbn)
        man_2d_corr_pred_attrbn = corr_pred_df['man_2d_feat_attrbn'].mean()
        man_2d_corr_pred_attrbn_lst.append(man_2d_corr_pred_attrbn)
        man_1d_corr_pred_attrbn = corr_pred_df['man_1d_feat_attrbn'].mean()
        man_1d_corr_pred_attrbn_lst.append(man_1d_corr_pred_attrbn)
        man_corr_pred_attrbn_lst.append(man_2d_corr_pred_attrbn + man_1d_corr_pred_attrbn)

        
        incorr_pred_df = spec_attrbn_df[spec_attrbn_df['pred_label'] != spec_attrbn_df['actual_label']]
        incorr_pred_df =  incorr_pred_df.reset_index(drop=True)
        tl_1d_incorr_pred_attrbn = incorr_pred_df['tl_1d_feat_attrbn'].mean()
        tl_1d_incorr_pred_attrbn_lst.append(tl_1d_incorr_pred_attrbn)
        man_2d_incorr_pred_attrbn = incorr_pred_df['man_2d_feat_attrbn'].mean()
        man_2d_incorr_pred_attrbn_lst.append(man_2d_incorr_pred_attrbn)
        man_1d_incorr_pred_attrbn = incorr_pred_df['man_1d_feat_attrbn'].mean()
        man_1d_incorr_pred_attrbn_lst.append(man_1d_incorr_pred_attrbn)
        man_incorr_pred_attrbn_lst.append(man_2d_incorr_pred_attrbn + man_1d_incorr_pred_attrbn)
    
    attribution_score_df = pd.DataFrame({
        'spec_type': spec_type_lst,
        'tl_1d_overall_attrbn': tl_1d_overall_attrbn_lst, 'tl_1d_corr_pred_attrbn': tl_1d_corr_pred_attrbn_lst, 'tl_1d_incorr_pred_attrbn': tl_1d_incorr_pred_attrbn_lst
        ,'man_2d_overall_attrbn': man_2d_overall_attrbn_lst, 'man_2d_corr_pred_attrbn': man_2d_corr_pred_attrbn_lst, 'man_2d_incorr_pred_attrbn': man_2d_incorr_pred_attrbn_lst
        ,'man_1d_overall_attrbn': man_1d_overall_attrbn_lst, 'man_1d_corr_pred_attrbn': man_1d_corr_pred_attrbn_lst, 'man_1d_incorr_pred_attrbn': man_1d_incorr_pred_attrbn_lst
        ,'man_overall_attrbn': man_overall_attrbn_lst, 'man_corr_pred_attrbn': man_corr_pred_attrbn_lst, 'man_incorr_pred_attrbn': man_incorr_pred_attrbn_lst
    })
    attribution_score_file_path = os.path.join(root_path, 'dataset/proc_data_DS/attribution_score.csv')
    attribution_score_df.to_csv(attribution_score_file_path, index=False)
    print('inside calc_attribution_score() method -end')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj')

    calc_attribution_score(root_path)