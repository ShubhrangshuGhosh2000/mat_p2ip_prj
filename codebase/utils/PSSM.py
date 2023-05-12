import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)
import torch
import joblib

from utils.PreprocessUtils import loadPSSM

def PSSM(fastas, directory, processPSSM=True,deviceType='cpu'):
    # ######### MUST RUN THE FOLLOWING IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH (see genPSSM() method of PreprocessUtils.py)
    # export PATH=$PATH:/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin
    # echo $PATH

    pssm_dict = {}
    #calculate the sum of all PSSM data
    # for item in fastas:
    for ind in range(len(fastas)):
        item = fastas[ind]
        name = item[0]
        print('@@@@@@@@@@@ ind: ' + str(ind))
        print('@@@@@@@@@@@ name: ' + str(name))
        seq = item[1]
        data = loadPSSM(name, seq, directory+'PSSM/',usePSIBlast=processPSSM)
        pssm_dict[name] = {# 'seq': seq, 
                           'pssm_val': torch.tensor(data)}
        print('\n\n################### Completed ' + str(ind+1) + ' out of ' + str(len(fastas)) + '\n\n')

    # save pssm_dict to a .pkl file
    print("\n Saving pssm_dict to a .pkl file...")
    filename = os.path.join(directory, 'pssm_dict.pkl')
    joblib.dump(value=pssm_dict, filename=filename, compress=3)
    print("\n The pssm_dict is saved as: " + filename)
