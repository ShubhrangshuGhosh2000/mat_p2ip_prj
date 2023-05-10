import sys, os
import pandas as pd
import joblib

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils_benchmark.ConjointTriad import ConjointTriad
from utils_benchmark.PreprocessUtils import readFasta
from utils_benchmark.PSEAAC import PSEAAC
from utils_benchmark.LDCTD import LDCTD
from utils_benchmark.AutoCovariance import AutoCovariance
from utils_benchmark.SkipGram import SkipGram
from utils_benchmark.OneHotEncoding import OneHotEncoding
from utils_benchmark.LabelEncoding import LabelEncoding
from utils_benchmark.PSSM import PSSM
from utils_benchmark.Blosum62 import Blosum62

import time


def createFeatures(folderName,featureSets,processPSSM=True,deviceType='cpu',spec_type=None):
    t =time.time()
    fastas = None
    if(spec_type is None):
        fastas = readFasta(folderName+'allSeqs.fasta')
    else:
        fastas = readFasta(folderName+spec_type+'.fasta')
    print('fasta loaded',time.time()-t)
    
    if 'AC30' in featureSets:
        #Guo AC calculation
        #calc AC
        ac = AutoCovariance(fastas,lag=30,deviceType=deviceType)

        f = open(folderName+'AC30.tsv','w')
        for item in ac:
            f.write('\t'.join(str(s) for s in item)+'\n')
        f.close()
        print('AC30',time.time()-t)


    if 'LD10_CTD' in featureSets:
        #Composition/Transition/Distribution Using Conjoint Triad Features on LD encoding
        (comp, tran, dist) = LDCTD(fastas)
        
        lst1 = [comp,tran,dist]
        lst2 = ['LD10_CTD_ConjointTriad_C.tsv','LD10_CTD_ConjointTriad_T.tsv','LD10_CTD_ConjointTriad_D.tsv']
        for i in range(0,3):
            f = open(folderName+lst2[i],'w')
            for item in lst1[i]:
                f.write('\t'.join(str(s) for s in item)+'\n')
            f.close()
        print('LD10_CTD',time.time()-t)


    if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
        #Li's PAAC used the first 3 variables from his moran AAC list, which appears to match what other authors have used
        paac = PSEAAC(fastas,lag=15)
        f = open(folderName+'PSAAC15.tsv','w')
        for item in paac:
            f.write('\t'.join(str(s) for s in item)+'\n')
        f.close()
        print('PSAAC15',time.time()-t)


    if 'conjointTriad' in featureSets or 'CT' in featureSets:
        #Conjoint Triad
        ct = ConjointTriad(fastas,deviceType=deviceType)
        f = open(folderName+'ConjointTriad.tsv','w')
        for item in ct:
            f.write('\t'.join(str(s) for s in item)+'\n')
        f.close()
        print('CT',time.time()-t)

    if 'SkipGramAA7' in featureSets:
        # SkipGram(fastas,folderName+'SkipGramAA7H5.encode',hiddenSize=5,deviceType='cuda',fullGPU=True)
        SkipGram(fastas,folderName+'SkipGramAA7H5.encode',hiddenSize=5,deviceType='cpu',fullGPU=False)
        print('SkipGramAA7',time.time()-t)

    if 'OneHotEncoding7' in featureSets:
        # For PIPR manual features in 2d format
        OneHotEncoding(fastas,folderName+'OneHotEncoding7.encode')
        print('OneHotEncoding7',time.time()-t)

    if 'LabelEncoding' in featureSets:
        LabelEncoding(fastas,folderName+'LabelEncoding.encode')
        print('LabelEncoding',time.time()-t)

    if 'PSSM' in featureSets:
        PSSM(fastas,folderName,processPSSM=True,deviceType='cpu')
        print('PSSM',time.time()-t)

    if 'Blosum62' in featureSets:
        Blosum62(fastas,folderName)
        print('Blosum62',time.time()-t) 


def createMixedFeatures(root_path,folderName,featureSets,processPSSM=True,deviceType='cpu'):
    t =time.time()
    # load the pkl file containing the tl-model extracted features for the proteins in a dictionary
    human_seq_feat_dict_tl = joblib.load(os.path.join(root_path, 'dataset/preproc_data','human_seq_feat_dict_prot_t5_xl_uniref50.pkl'))
    
    if 'AC30' in featureSets:
        print('Creating the mixed features combination for AC30')
        # retrieve the manual features
        ac_df_man = pd.read_csv(filepath_or_buffer=os.path.join(folderName, 'AC30.tsv'), sep='\t')
        # prepare the tl-based features by iterating through ac_df_man
        ac_lst_tl = []
        for index, row in ac_df_man.iterrows():
            # extract tl based features for the given protein id
            prot_id = row[0]
            prot_feat_arr_tl = human_seq_feat_dict_tl[prot_id]['seq_feat']
            ac_lst_tl.append(prot_feat_arr_tl)
        # create ac_df_tl from ac_lst_tl
        col_nm_lst = ['tl_feat_' + str(idx) for idx in range(0, 1024)]
        ac_df_tl = pd.DataFrame(data=ac_lst_tl, columns=col_nm_lst)
        # concatenate both the data-frames (ac_df_man, ac_df_tl) horizontally
        ac_df_mixed = pd.concat([ac_df_man, ac_df_tl], axis=1, sort=False)
        # save ac_df_mixed
        ac_df_mixed.to_csv(os.path.join(folderName, 'AC30_mixed.tsv'), sep='\t', index=False)
        print('AC30',time.time()-t)


    if 'LD10_CTD' in featureSets:
        print('Creating the mixed features combination for LD10_CTD')
        # retrieve the manual features
        ld10_ctd_df_man = pd.read_csv(filepath_or_buffer=os.path.join(folderName, 'LD10_CTD_ConjointTriad_D.tsv'), sep='\t')
        # prepare the tl-based features by iterating through ld10_ctd_df_man
        ld10_ctd_lst_tl = []
        for index, row in ld10_ctd_df_man.iterrows():
            # extract tl based features for the given protein id
            prot_id = row[0]
            prot_feat_arr_tl = human_seq_feat_dict_tl[prot_id]['seq_feat']
            ld10_ctd_lst_tl.append(prot_feat_arr_tl)
        # create ld10_ctd_df_tl from ld10_ctd_lst_tl
        col_nm_lst = ['tl_feat_' + str(idx) for idx in range(0, 1024)]
        ld10_ctd_df_tl = pd.DataFrame(data=ld10_ctd_lst_tl, columns=col_nm_lst)
        # concatenate both the data-frames (ld10_ctd_df_man, ld10_ctd_df_tl) horizontally
        ld10_ctd_df_mixed = pd.concat([ld10_ctd_df_man, ld10_ctd_df_tl], axis=1, sort=False)
        # save ld10_ctd_df_mixed
        ld10_ctd_df_mixed.to_csv(os.path.join(folderName, 'LD10_CTD_ConjointTriad_D_mixed.tsv'), sep='\t', index=False)
        print('LD10_CTD',time.time()-t)


    if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
        print('Creating the mixed features combination for PSAAC15')
        # retrieve the manual features
        psaac_df_man = pd.read_csv(filepath_or_buffer=os.path.join(folderName, 'PSAAC15.tsv'), sep='\t')
        # prepare the tl-based features by iterating through psaac_df_man
        psaac_lst_tl = []
        for index, row in psaac_df_man.iterrows():
            # extract tl based features for the given protein id
            prot_id = row[0]
            prot_feat_arr_tl = human_seq_feat_dict_tl[prot_id]['seq_feat']
            psaac_lst_tl.append(prot_feat_arr_tl)
        # create psaac_df_tl from psaac_lst_tl
        col_nm_lst = ['tl_feat_' + str(idx) for idx in range(0, 1024)]
        psaac_df_tl = pd.DataFrame(data=psaac_lst_tl, columns=col_nm_lst)
        # concatenate both the data-frames (psaac_df_man, psaac_df_tl) horizontally
        psaac_df_mixed = pd.concat([psaac_df_man, psaac_df_tl], axis=1, sort=False)
        # save psaac_df_mixed
        psaac_df_mixed.to_csv(os.path.join(folderName, 'PSAAC15_mixed.tsv'), sep='\t', index=False)
        print('PSAAC15',time.time()-t)


    if 'conjointTriad' in featureSets or 'CT' in featureSets:
        print('Creating the mixed features combination for ConjointTriad')
        # retrieve the manual features
        ct_df_man = pd.read_csv(filepath_or_buffer=os.path.join(folderName, 'ConjointTriad.tsv'), sep='\t')
        # prepare the tl-based features by iterating through ct_df_man
        ct_lst_tl = []
        for index, row in ct_df_man.iterrows():
            # extract tl based features for the given protein id
            prot_id = row[0]
            prot_feat_arr_tl = human_seq_feat_dict_tl[prot_id]['seq_feat']
            ct_lst_tl.append(prot_feat_arr_tl)
        # create ct_df_tl from ct_lst_tl
        col_nm_lst = ['tl_feat_' + str(idx) for idx in range(0, 1024)]
        ct_df_tl = pd.DataFrame(data=ct_lst_tl, columns=col_nm_lst)
        # concatenate both the data-frames (ct_df_man, ct_df_tl) horizontally
        ct_df_mixed = pd.concat([ct_df_man, ct_df_tl], axis=1, sort=False)
        # save ct_df_mixed
        ct_df_mixed.to_csv(os.path.join(folderName, 'ConjointTriad_mixed.tsv'), sep='\t', index=False)
        print('ConjointTriad',time.time()-t)
