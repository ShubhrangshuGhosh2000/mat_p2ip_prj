import sys, os
# currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(currentdir)

# import numpy as np
# #add parent to path
# currentdir = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(currentdir)
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)
import torch
import joblib

from utils_benchmark.PreprocessUtils import calcBlosum62

def Blosum62(fastas, directory):
    # ######### MUST RUN THE FOLLOWING IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH (see genBlosum62() method of PreprocessUtils.py)
    # export PATH=$PATH:/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin
    # echo $PATH

    blosum62_dict = {}
    # for item in fastas:
    for ind in range(len(fastas)):
    # for ind in range(13300, len(fastas)):
        item = fastas[ind]
        name = item[0]
        seq = item[1]
        data = calcBlosum62(name, seq)
        blosum62_dict[name] = {# 'seq': seq, 
                           'blosum62_val': torch.tensor(data)}
        print('\n\n################### Completed ' + str(ind+1) + ' out of ' + str(len(fastas)) + '\n\n')

    # save blosum62_dict to a .pkl file
    print("\n Saving blosum62_dict to a .pkl file...")
    filename = os.path.join(directory, 'blosum62_dict.pkl')
    joblib.dump(value=blosum62_dict, filename=filename, compress=3)
    print("\n The blosum62_dict is saved as: " + filename)
