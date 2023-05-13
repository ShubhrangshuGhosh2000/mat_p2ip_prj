import sys, os

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import statistics
import pandas as pd
from utils.ProjectDataLoader import *


# root_path = os.path.join('/home/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
# root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
resultsFolderName = os.path.join(root_path, 'dataset/proc_data/benchmark_res/piprp_res/piprp_res_origMan_auxTlOtherMan_xai/')


def calc_score_for_HumanRandom50():
    print('inside calc_score_for_HumanRandom50() method -start')
    pred_attrbn_fnames_lst = []
    for i in range(0,5):
        pred_attrbn_fnames_lst.append(resultsFolderName+'R50_'+str(i)+'_predict_attrbn.tsv')
    attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst)
    # save attribution_score_df
    attribution_score_df.to_csv(os.path.join(resultsFolderName, 'R50_attribution_score.csv'), index=False)
    print('inside calc_score_for_HumanRandom50() method -end')


def calc_score_for_HumanRandom20():
    print('inside calc_score_for_HumanRandom20() method -start')
    pred_attrbn_fnames_lst = [[],[]]
    for i in range(0,5):
        pred_attrbn_fnames_lst[0].append(resultsFolderName+'R20_'+str(i)+'_predict1_attrbn.tsv')
        pred_attrbn_fnames_lst[1].append(resultsFolderName+'R20_'+str(i)+'_predict2_attrbn.tsv')
    predict1_attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst[0])
    predict2_attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst[1])

    # save attribution_score_df
    predict1_attribution_score_df.to_csv(os.path.join(resultsFolderName, 'R20_predict1_attribution_score.csv'), index=False)
    predict2_attribution_score_df.to_csv(os.path.join(resultsFolderName, 'R20_predict2_attribution_score.csv'), index=False)
    print('inside calc_score_for_HumanRandom20() method -end')


def calc_score_for_HumanHeldOut50():
    print('inside calc_score_for_HumanHeldOut50() method -start')
    pred_attrbn_fnames_lst = []
    for i in range(0,6):
        for j in range(i,6):
            pred_attrbn_fnames_lst.append(resultsFolderName+'H50_'+str(i)+'_'+str(j)+'_predict_attrbn.tsv')
    attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst)
    # save attribution_score_df
    attribution_score_df.to_csv(os.path.join(resultsFolderName, 'H50_attribution_score.csv'), index=False)
    print('inside calc_score_for_HumanHeldOut50() method -end')


def calc_score_for_HumanHeldOut20():
    print('inside calc_score_for_HumanHeldOut20() method -start')
    pred_attrbn_fnames_lst = [[],[]]
    for i in range(0,6):
        for j in range(i,6):
            pred_attrbn_fnames_lst[0].append(resultsFolderName+'H20_'+str(i)+'_'+str(j)+'_predict1_attrbn.tsv')
            pred_attrbn_fnames_lst[1].append(resultsFolderName+'H20_'+str(i)+'_'+str(j)+'_predict2_attrbn.tsv')
    predict1_attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst[0])
    predict2_attribution_score_df = calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst[1])

    # save attribution_score_df
    predict1_attribution_score_df.to_csv(os.path.join(resultsFolderName, 'H20_predict1_attribution_score.csv'), index=False)
    predict2_attribution_score_df.to_csv(os.path.join(resultsFolderName, 'H20_predict2_attribution_score.csv'), index=False)
    print('inside calc_score_for_HumanHeldOut20() method -end')


def calc_score_for_all():
    print('inside calc_score_for_all() method -start')
    # for R50
    R50_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'R50_attribution_score.csv'))
    avg_R50_attribution_score_df = R50_attribution_score_df[R50_attribution_score_df['file_name'] == 'AVG']
    avg_R50_attribution_score_df['file_name'] = ['R50']

    # for R20
    R20_predict1_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'R20_predict1_attribution_score.csv'))
    avg_R20_predict1_attribution_score_df = R20_predict1_attribution_score_df[R20_predict1_attribution_score_df['file_name'] == 'AVG']
    avg_R20_predict1_attribution_score_df['file_name'] = ['R20_predict1']
    R20_predict2_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'R20_predict2_attribution_score.csv'))
    avg_R20_predict2_attribution_score_df = R20_predict2_attribution_score_df[R20_predict2_attribution_score_df['file_name'] == 'AVG']
    avg_R20_predict2_attribution_score_df['file_name'] = ['R20_predict2']

    # for H50
    H50_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'H50_attribution_score.csv'))
    avg_H50_attribution_score_df = H50_attribution_score_df[H50_attribution_score_df['file_name'] == 'AVG']
    avg_H50_attribution_score_df['file_name'] = ['H50']

    # for H20
    H20_predict1_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'H20_predict1_attribution_score.csv'))
    avg_H20_predict1_attribution_score_df = H20_predict1_attribution_score_df[H20_predict1_attribution_score_df['file_name'] == 'AVG']
    avg_H20_predict1_attribution_score_df['file_name'] = ['H20_predict1']
    H20_predict2_attribution_score_df = pd.read_csv(os.path.join(resultsFolderName, 'H20_predict2_attribution_score.csv'))
    avg_H20_predict2_attribution_score_df = H20_predict2_attribution_score_df[H20_predict2_attribution_score_df['file_name'] == 'AVG']
    avg_H20_predict2_attribution_score_df['file_name'] = ['H20_predict2']

    # concat all the avg dataframes together row-wise
    attribn_score_for_all_df = pd.concat([avg_R50_attribution_score_df, avg_R20_predict1_attribution_score_df, avg_R20_predict2_attribution_score_df
                                            , avg_H50_attribution_score_df, avg_H20_predict1_attribution_score_df, avg_H20_predict2_attribution_score_df], axis=0)

    # save attribn_score_for_all_df
    attribn_score_for_all_df.to_csv(os.path.join(resultsFolderName, 'attribn_score_for_all.csv'), index=False)
    print('inside calc_score_for_all() method -end')



# calculate aggregated attribution
def calc_aggrg_pred_attrbn(pred_attrbn_fnames_lst=None):
    print('inside calc_aggrg_pred_attrbn() method -start')
    pred_fnames_lst = []
    tl_1d_overall_attrbn_lst, tl_1d_corr_pred_attrbn_lst, tl_1d_incorr_pred_attrbn_lst = [], [], []
    man_2d_overall_attrbn_lst, man_2d_corr_pred_attrbn_lst, man_2d_incorr_pred_attrbn_lst = [], [], []
    man_1d_overall_attrbn_lst, man_1d_corr_pred_attrbn_lst, man_1d_incorr_pred_attrbn_lst = [], [], []
    man_overall_attrbn_lst, man_corr_pred_attrbn_lst, man_incorr_pred_attrbn_lst = [], [], []

    for pred_attrbn_fname in pred_attrbn_fnames_lst:
        print('\n########## pred_attrbn_fname: ' + str(pred_attrbn_fname))
        specific_attrbn_df = pd.read_csv(pred_attrbn_fname)
        pred_fnames_lst.append(pred_attrbn_fname.replace(resultsFolderName, '').replace('_attrbn.tsv', ''))

        # calculate overall avg. attribution
        tl_1d_overall_attrbn = specific_attrbn_df['tl_1d_feat_attrbn'].mean()
        tl_1d_overall_attrbn_lst.append(tl_1d_overall_attrbn)
        man_2d_overall_attrbn = specific_attrbn_df['man_2d_feat_attrbn'].mean()
        man_2d_overall_attrbn_lst.append(man_2d_overall_attrbn)
        man_1d_overall_attrbn = specific_attrbn_df['man_1d_feat_attrbn'].mean()
        man_1d_overall_attrbn_lst.append(man_1d_overall_attrbn)
        man_overall_attrbn_lst.append(man_2d_overall_attrbn + man_1d_overall_attrbn)

        # calculate avg. attribution for the correct predictions
        corr_pred_df = specific_attrbn_df[specific_attrbn_df['pred_label'] == specific_attrbn_df['actual_label']]
        corr_pred_df =  corr_pred_df.reset_index(drop=True)
        tl_1d_corr_pred_attrbn = corr_pred_df['tl_1d_feat_attrbn'].mean()
        tl_1d_corr_pred_attrbn_lst.append(tl_1d_corr_pred_attrbn)
        man_2d_corr_pred_attrbn = corr_pred_df['man_2d_feat_attrbn'].mean()
        man_2d_corr_pred_attrbn_lst.append(man_2d_corr_pred_attrbn)
        man_1d_corr_pred_attrbn = corr_pred_df['man_1d_feat_attrbn'].mean()
        man_1d_corr_pred_attrbn_lst.append(man_1d_corr_pred_attrbn)
        man_corr_pred_attrbn_lst.append(man_2d_corr_pred_attrbn + man_1d_corr_pred_attrbn)

        # calculate avg. attribution for the incorrect predictions
        incorr_pred_df = specific_attrbn_df[specific_attrbn_df['pred_label'] != specific_attrbn_df['actual_label']]
        incorr_pred_df =  incorr_pred_df.reset_index(drop=True)
        tl_1d_incorr_pred_attrbn = incorr_pred_df['tl_1d_feat_attrbn'].mean()
        tl_1d_incorr_pred_attrbn_lst.append(tl_1d_incorr_pred_attrbn)
        man_2d_incorr_pred_attrbn = incorr_pred_df['man_2d_feat_attrbn'].mean()
        man_2d_incorr_pred_attrbn_lst.append(man_2d_incorr_pred_attrbn)
        man_1d_incorr_pred_attrbn = incorr_pred_df['man_1d_feat_attrbn'].mean()
        man_1d_incorr_pred_attrbn_lst.append(man_1d_incorr_pred_attrbn)
        man_incorr_pred_attrbn_lst.append(man_2d_incorr_pred_attrbn + man_1d_incorr_pred_attrbn)
    # end of for loop: for pred_attrbn_fname in pred_attrbn_fnames_lst:

    # calculate the overall average attribution
    pred_fnames_lst.append('AVG')
    tl_1d_overall_attrbn_lst.append(statistics.fmean(tl_1d_overall_attrbn_lst))
    tl_1d_corr_pred_attrbn_lst.append(statistics.fmean(tl_1d_corr_pred_attrbn_lst))
    tl_1d_incorr_pred_attrbn_lst.append(statistics.fmean(tl_1d_incorr_pred_attrbn_lst))

    man_2d_overall_attrbn_lst.append(statistics.fmean(man_2d_overall_attrbn_lst))
    man_2d_corr_pred_attrbn_lst.append(statistics.fmean(man_2d_corr_pred_attrbn_lst))
    man_2d_incorr_pred_attrbn_lst.append(statistics.fmean(man_2d_incorr_pred_attrbn_lst))

    man_1d_overall_attrbn_lst.append(statistics.fmean(man_1d_overall_attrbn_lst))
    man_1d_corr_pred_attrbn_lst.append(statistics.fmean(man_1d_corr_pred_attrbn_lst))
    man_1d_incorr_pred_attrbn_lst.append(statistics.fmean(man_1d_incorr_pred_attrbn_lst))

    man_overall_attrbn_lst.append(statistics.fmean(man_overall_attrbn_lst))
    man_corr_pred_attrbn_lst.append(statistics.fmean(man_corr_pred_attrbn_lst))
    man_incorr_pred_attrbn_lst.append(statistics.fmean(man_incorr_pred_attrbn_lst))

    # create attribution_score_df and save it
    attribution_score_df = pd.DataFrame({
        'file_name': pred_fnames_lst,
        'tl_1d_overall_attrbn': tl_1d_overall_attrbn_lst, 'tl_1d_corr_pred_attrbn': tl_1d_corr_pred_attrbn_lst, 'tl_1d_incorr_pred_attrbn': tl_1d_incorr_pred_attrbn_lst
        ,'man_2d_overall_attrbn': man_2d_overall_attrbn_lst, 'man_2d_corr_pred_attrbn': man_2d_corr_pred_attrbn_lst, 'man_2d_incorr_pred_attrbn': man_2d_incorr_pred_attrbn_lst
        ,'man_1d_overall_attrbn': man_1d_overall_attrbn_lst, 'man_1d_corr_pred_attrbn': man_1d_corr_pred_attrbn_lst, 'man_1d_incorr_pred_attrbn': man_1d_incorr_pred_attrbn_lst
        ,'man_overall_attrbn': man_overall_attrbn_lst, 'man_corr_pred_attrbn': man_corr_pred_attrbn_lst, 'man_incorr_pred_attrbn': man_incorr_pred_attrbn_lst
    })
    print('inside calc_aggrg_pred_attrbn() method -end')
    return attribution_score_df


def calc_score_for_specific_ds_genre(ds_genre = 'HumanRandom50'):
    print('\ninside calc_score_for_specific_ds_genre() method -start')
    print('\n########## ds_genre: ' + ds_genre)
    if(ds_genre == 'HumanRandom50'):
        calc_score_for_HumanRandom50()
    elif(ds_genre == 'HumanRandom20'):
        calc_score_for_HumanRandom20()
    elif(ds_genre == 'HumanHeldOut50'):
        calc_score_for_HumanHeldOut50()
    elif(ds_genre == 'HumanHeldOut20'):
        calc_score_for_HumanHeldOut20()
    elif(ds_genre == 'all'):
        calc_score_for_all()
    print('inside calc_score_for_specific_ds_genre() method -end')


if __name__ == '__main__':
    # human_ds_genres = ['HumanRandom50', 'HumanRandom20', 'HumanHeldOut50', 'HumanHeldOut20', 'all']
    human_ds_genres = ['all']

    for ds_genre in human_ds_genres:
        calc_score_for_specific_ds_genre(ds_genre)
