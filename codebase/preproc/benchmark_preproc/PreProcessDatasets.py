import sys

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
        SkipGram(fastas,folderName+'SkipGramAA7H5.encode',hiddenSize=5,deviceType='cpu',fullGPU=False)
        print('SkipGramAA7',time.time()-t)

    if 'OneHotEncoding7' in featureSets:
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
