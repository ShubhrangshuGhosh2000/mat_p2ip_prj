# this is the main module for manual feature extraction/engineering
import numpy as np
import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from preproc.feat_engg_manual.AutoCovariance import AutoCovariance
from preproc.feat_engg_manual.ConjointTriad import ConjointTriad
from preproc.feat_engg_manual.LDCTD import LDCTD
from preproc.feat_engg_manual.PSEAAC import PSEAAC
from preproc.feat_engg_manual.AAC import AAC
from preproc.feat_engg_manual.Chaos import Chaos
from  preproc.feat_engg_manual.CTD_Composition import CTD_Composition
from  preproc.feat_engg_manual.CTD_Distribution import CTD_Distribution
from  preproc.feat_engg_manual.CTD_Transition import CTD_Transition
from preproc.feat_engg_manual.PairwiseDist import PairwiseDist
from preproc.feat_engg_manual.QuasiSequenceOrder import QuasiSequenceOrder


OUTPUT_DIR = 'dataset/preproc_data/human_2021/manual_feat'

def extract_prot_seq_manual_feat(root_path='./', prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE', feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD'], deviceType='cpu'):
    print('#### inside the extract_prot_seq_manual_feat() method - Start')
    fastas = [('999', prot_seq)]
    featureSets = set(feature_type_lst)
    # the dictionary to be returned
    seq_manual_feat_dict = {}

    if 'AC30' in featureSets:
        print("Calculating 'AC30' feature - Start")
        #Guo AC calculation
        #calc AC
        ac = AutoCovariance(fastas, lag=30, deviceType=deviceType)    
        seq_manual_feat_dict['AC30'] = ac[1][1:]
        print("Calculating 'AC30' feature - End")

    if 'PSAAC15' in featureSets or 'PSEAAC15' in featureSets:
        print("Calculating 'PSAAC15' feature - Start")
        #Li's PAAC used the first 3 variables from his moran AAC list, which appears to match what other authors have used
        paac = PSEAAC(fastas,lag=15)
        seq_manual_feat_dict['PSAAC15'] = paac[1][1:]
        print("Calculating 'PSAAC15' feature - End")

    if 'ConjointTriad' in featureSets or 'CT' in featureSets:
        print("Calculating 'ConjointTriad' feature - Start")
        #Conjoint Triad
        ct = ConjointTriad(fastas,deviceType=deviceType)
        seq_manual_feat_dict['ConjointTriad'] = ct[1][1:]
        print("Calculating 'ConjointTriad' feature - End")

    if 'LD10_CTD' in featureSets:
        print("Calculating 'LD10_CTD' feature - Start")
        #Composition/Transition/Distribution Using Conjoint Triad Features on LD encoding
        (comp, tran, dist) = LDCTD(fastas)
        # lst1 = [comp,tran,dist]
        # lst2 = ['LD10_CTD_ConjointTriad_C.tsv','LD10_CTD_ConjointTriad_T.tsv','LD10_CTD_ConjointTriad_D.tsv']
        # for i in range(0,3):
        #     f = open(folderName+lst2[i],'w')
        #     for item in lst1[i]:
        #         f.write('\t'.join(str(s) for s in item)+'\n')
        #     f.close()
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_C'] = comp[1][1:]
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_T'] = tran[1][1:]
        seq_manual_feat_dict['LD10_CTD_ConjointTriad_D'] = dist[1][1:]
        print("Calculating 'LD10_CTD' feature - End")

    if 'CHAOS' in featureSets:
        print("Calculating 'CHAOS' feature - Start")
        vals = Chaos(fastas)
        seq_manual_feat_dict['CHAOS'] = vals[1][1:]
        print("Calculating 'CHAOS' feature - End")

    if 'AAC20' in featureSets:
        print("Calculating 'AAC20' feature - Start")
        aac = AAC(fastas,deviceType=deviceType)
        seq_manual_feat_dict['AAC20'] = aac[1][1:]
        print("Calculating 'AAC20' feature - End")

    if 'AAC400' in featureSets:
        print("Calculating 'AAC400' feature - Start")
        aac = AAC(fastas, groupLen=2, deviceType=deviceType)
        seq_manual_feat_dict['AAC400'] = aac[1][1:]
        print("Calculating 'AAC400' feature - End")

    if 'DUMULTIGROUPCTD' in featureSets:
        print("Calculating 'DUMULTIGROUPCTD' feature - Start")
        groupings = {}
        groupings['Hydrophobicity'] = ['RKEDQN','GASTPHY','CLVIMFW']
        groupings['Normalized_van_der_Waals_volume'] = ['GASTPD','NVEQIL','MHKFRYW']
        groupings['Polarity'] = ['LIFWCMVY','PATGS','HQRKNED']
        groupings['Polarizability'] = ['GASDT','CPNVEQIL','KMHFRYW']
        groupings['Charge'] = ['KR','ANCQGHILMFPSTWYV','DE']
        groupings['Secondary_structure'] = ['EALMQKRH','VIYCWFT','GNPSD']
        groupings['Solvent_accessibility'] = ['ALFCGIVW','PKQEND','MPSTHY']
        groupings['Surface_tension'] = ['GQDNAHR','KTSEC','ILMFPWYV']
        groupings['Protein-protein_interface_hotspot_propensity-Bogan'] = ['DHIKNPRWY','EQSTGAMF','CLV']
        groupings['Protein-protein_interface_propensity-Ma'] = ['CDFMPQRWY','AGHVLNST','EIK']
        groupings['Protein-DNA_interface_propensity-Schneider'] = ['GKNQRSTY','ADEFHILVW','CMP']
        groupings['Protein-DNA_interface_propensity-Ahmad'] = ['GHKNQRSTY','ADEFIPVW','CLM']
        groupings['Protein-RNA_interface_propensity-Kim'] = ['HKMRY','FGILNPQSVW','CDEAT']
        groupings['Protein-RNA_interface_propensity-Ellis'] = ['HGKMRSYW','AFINPQT','CDELV']
        groupings['Protein-RNA_interface_propensity-Phipps'] = ['HKMQRS','ADEFGLNPVY','CITW']
        groupings['Protein-ligand_binding_site_propensity_-Khazanov'] = ['CFHWY','GILNMSTR','AEDKPQV']
        groupings['Protein-ligand_valid_binding_site_propensity_-Khazanov'] = ['CFHWYM','DGILNSTV','AEKPQR']
        groupings['Propensity_for_protein-ligand_polar_&_aromatic_non-bonded_interactions-Imai'] = ['DEHRY','CFKMNQSTW','AGILPV']
        groupings['Molecular_Weight'] = ['AGS','CDEHIKLMNQPTV','FRWY']
        groupings['cLogP'] = ['RKDNEQH','PYSTGACV','WMFLI']
        groupings['No_of_hydrogen_bond_donor_in_side_chain'] = ['HKNQR','DESTWY','ACGFILMPV']
        groupings['No_of_hydrogen_bond_acceptor_in_side_chain'] = ['DEHNQR','KSTWY','ACGFILMPV']
        groupings['Solubility_in_water'] = ['ACGKRT','EFHILMNPQSVW','DY']
        groupings['Amino_acid_flexibility_index'] = ['EGKNQS','ADHIPRTV','CFLMWY']
        for item in [(CTD_Composition,'DuMultiCTD_C'),(CTD_Transition,'DuMultiCTD_T'),(CTD_Distribution,'DuMultiCTD_D')]:
            func = item[0]
            vals = []
            for feat in groupings:
                results = func(fastas,groupings=groupings[feat])
                for i in range(0,len(results[0])): #header row
                    results[0][i] = feat+'_'+results[0][i] #add feature name to each column in header row
                results = np.asarray(results)
                if len(vals) == 0:
                    vals.append(results)
                else:
                    vals.append(results[:,1:])#remove protein names if not first group calculated
            vals = np.hstack(vals).tolist()

            # f = open(folderName+item[1]+'.tsv','w')
            # for line in vals:
            #     f.write('\t'.join(str(s) for s in line)+'\n')
            # f.close()
            seq_manual_feat_dict[item[1]] = vals[1][1:]
        # end of for loop
        print("Calculating 'DUMULTIGROUPCTD' feature - End")

    if 'Grantham_Sequence_Order_30' in featureSets:
        print("Calculating 'Grantham_Sequence_Order_30' feature - Start")
        ac = PairwiseDist(fastas, pairwiseAAIDs=['Grantham'], calcType='SumSq', lag=30)
        seq_manual_feat_dict['Grantham_Sequence_Order_30'] = ac[1][1:]
        print("Calculating 'Grantham_Sequence_Order_30' feature - End")

    if 'Schneider_Sequence_Order_30' in featureSets:
        print("Calculating 'Schneider_Sequence_Order_30' feature - Start")
        ac = PairwiseDist(fastas, pairwiseAAIDs=['Schneider-Wrede'],calcType='SumSq', lag=30)
        seq_manual_feat_dict['Schneider_Sequence_Order_30'] = ac[1][1:]
        print("Calculating 'Schneider_Sequence_Order_30' feature - End")

    if 'Grantham_Quasi_30' in featureSets:
        print("Calculating 'Grantham_Quasi_30' feature - Start")
        paac = QuasiSequenceOrder(fastas, pairwiseAAIDs=['Grantham'],lag=30)
        seq_manual_feat_dict['Grantham_Quasi_30'] = ac[1][1:]
        print("Calculating 'Grantham_Quasi_30' feature - End")

    if 'Schneider_Quasi_30' in featureSets:
        print("Calculating 'Schneider_Quasi_30' feature - Start")
        paac = QuasiSequenceOrder(fastas, pairwiseAAIDs=['Schneider-Wrede'],lag=30)
        seq_manual_feat_dict['Schneider_Quasi_30'] = ac[1][1:]
        print("Calculating 'Schneider_Quasi_30' feature - End")

    if 'APSAAC30_2' in featureSets:
        print("Calculating 'APSAAC30_2' feature - Start")
        #Note, Du's paper states W=0.5, but the standard is W=0.05.  In theory, W should be lower with more attributes (due to amphipathic=True) to balance betters with AA counts.
        #Currently leaving at 0.05, assuming this may be a typo.  Can change later as necessary.
        paac = PSEAAC(fastas,aaIDs=['GUO_H1','HOPT810101'],lag=30,amphipathic=True)
        seq_manual_feat_dict['APSAAC30_2'] = paac[1][1:]
        print("Calculating 'APSAAC30_2' feature - End")

    print('#### inside the extract_prot_seq_manual_feat() method - End')
    return seq_manual_feat_dict


if __name__ == '__main__':
    # root_path = os.path.join('/home/sg/Shubh_Working_Ubuntu/Workspaces/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    # root_path = os.path.join('/home/rs/19CS92W02/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/only_seq_prj_v1')

    prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE'
    # The feature mappings are:
    # 'AAC' ==> AC30 (For Li2020)
    # 'PAAC' ==> PSAAC15 (For Li2020)
    # 'CT' ==> ConjointTriad (For Li2020)
    # 'LD' ==> LD10_CTD (For Li2020)
    # 'CHAOS' ==> CHAOS (For Jia2019)
    # 'AAC' ==> AAC20 (For Jia2019, Du2017)
    # 'AAC400'  (For Du2017)
    # 'DUMULTIGROUPCTD' ==> 'DuMultiCTD_C','DuMultiCTD_D','DuMultiCTD_T' (For Du2017)
    # 'Grantham_Sequence_Order_30','Schneider_Sequence_Order_30' (For Du2017)
    # 'Grantham_Quasi_30','Schneider_Quasi_30' (For Du2017)
    # 'APSAAC30' ==> 'APSAAC30_2' (For Du2017)

    feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD', \
                        'CHAOS', \
                        'AAC20', 'AAC400', 'Grantham_Sequence_Order_30', 'Schneider_Sequence_Order_30', \
                        # 'DUMULTIGROUPCTD', \  # making the number of manual features too large
                        'Grantham_Quasi_30', 'Schneider_Quasi_30', 'APSAAC30_2']
    seq_manual_feat_dict = extract_prot_seq_manual_feat(root_path, prot_seq = prot_seq, feature_type_lst = feature_type_lst, deviceType='cpu')
