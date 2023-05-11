from utils_benchmark import PPIPUtils
import os
import sys

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'mtf_p2ip_prj' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import numpy as np

currentDir = os.path.join(path_root, 'dataset/preproc_data/benchmark_feat/')

#if dirLst, just get directory names, not data sets.  Used for pairwise predictors
def loadHumanRandom50(directory,augment = False,dirLst = False):
    trainSets = []
    testSets = []
    saves = []
    predFNames = []
    for i in range(0,5):
        if not dirLst:
            trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random50/Train_'+str(i)+'.tsv','int')))
            testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random50/Test_'+str(i)+'.tsv','int')))
        saves.append(directory+'R50_'+str(i)+'.out')
        predFNames.append(directory+'R50_'+str(i)+'_predict.tsv')
        
    if augment:
        trainSets, testSets = augmentAll(trainSets,testSets)
    # featDir = 'PPI_Datasets/Human2021/'
    featDir = os.path.join(path_root,'dataset/preproc_data/benchmark_feat/PPI_Datasets/Human2021/')
    if dirLst:
        featDir = []
        for i in range(0,5):
            featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/r50_'+str(i)+'/')
    return trainSets,testSets, saves,predFNames, featDir


def loadHumanRandom20(directory,augment = False,dirLst = False):
    trainSets = []
    testSets = []
    testSets2 = []
    saves = []
    predFNames = [[],[]]
    for i in range(0,5):
        if not dirLst:
            trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Train_'+str(i)+'.tsv','int')))
            testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Test1_'+str(i)+'.tsv','int')))
            testSets2.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/Random20/Test2_'+str(i)+'.tsv','int')))
        saves.append(directory+'R20_'+str(i)+'.out')
        predFNames[0].append(directory+'R20_'+str(i)+'_predict1.tsv')
        predFNames[1].append(directory+'R20_'+str(i)+'_predict2.tsv')
    if augment:
        trainSets, testSets = augmentAll(trainSets,testSets)
    featDir = currentDir+'PPI_Datasets/Human2021/'
    if dirLst:
        featDir = []
        newSaves = []
        for j in range(1,3):
            for i in range(0,5):
                featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/r20_'+str(j)+'_'+str(i)+'/')
                name = saves[i].split('.')
                name = name[0]+'_'+str(j)+name[1]
                newSaves.append(name)
            
        predFNames = predFNames[0] + predFNames[1]
        saves = newSaves
    return trainSets,[testSets,testSets2], saves, predFNames, featDir


def loadHumanHeldOut50(directory,augment = False, dirLst=False):
    trainSets = []
    testSets = []
    saves = []
    predFNames = []
    for i in range(0,6):
        for j in range(i,6):
            if not dirLst:
                trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut50/Train_'+str(i)+'_'+str(j)+'.tsv','int')))
                testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut50/Test_'+str(i)+'_'+str(j)+'.tsv','int')))
            saves.append(directory+'H50_'+str(i)+'_'+str(j)+'.out')
            predFNames.append(directory+'H50_'+str(i)+'_'+str(j)+'_predict.tsv')
    if augment:
        trainSets, testSets = augmentAll(trainSets,testSets)
    
    featDir = currentDir+'PPI_Datasets/Human2021/'
    if dirLst:
        featDir = []
        for i in range(0,6):
            for j in range(i,6):
                featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/h50_'+str(i)+'_'+str(j)+'/')
    return trainSets,testSets, saves,predFNames, featDir


def loadHumanHeldOut20(directory,augment = False, dirLst = False):
    trainSets = []
    testSets = []
    testSets2 = []
    saves = []
    predFNames = [[],[]]
    for i in range(0,6):
        for j in range(i,6):
            if not dirLst:
                trainSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Train_'+str(i)+'_'+str(j)+'.tsv','int')))
                testSets.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Test1_'+str(i)+'_'+str(j)+'.tsv','int')))
                testSets2.append(np.asarray(PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Human2021/BioGRID2021/HeldOut20/Test2_'+str(i)+'_'+str(j)+'.tsv','int')))
            saves.append(directory+'H20_'+str(i)+'_'+str(j)+'.out')
            predFNames[0].append(directory+'H20_'+str(i)+'_'+str(j)+'_predict1.tsv')
            predFNames[1].append(directory+'H20_'+str(i)+'_'+str(j)+'_predict2.tsv')
    if augment:
        trainSets, testSets = augmentAll(trainSets,testSets)
    featDir = currentDir+'PPI_Datasets/Human2021/'
    if dirLst:
        featDir = []
        newSaves = []
        for k in range(1,3):
            idx = 0
            for i in range(0,6):
                for j in range(i,6):
                    featDir.append(currentDir+'PPI_Datasets/Human2021/BioGRID2021/pairwiseDatasets/h20_'+str(k)+'_'+str(i)+'_'+str(j)+'/')
                    name = saves[idx].split('.')
                    name = name[0]+'_'+str(k)+name[1]
                    newSaves.append(name)
                    idx += 1
            
        predFNames = predFNames[0] + predFNames[1]
        saves = saves + saves
    return trainSets,[testSets,testSets2], saves,predFNames, featDir


def loadDscriptData_human_full(resultsFolderName):
    featureFolder = os.path.join(path_root, 'dataset/preproc_data_DS/benchmark_feat/human/')
    trainSets  = []
    testSets = []
    saves = []
    predFNames = []
    trainSets = PPIPUtils.parseTSV(featureFolder+'human_full.tsv','string')
    trainSets = [np.asarray(trainSets)]
    saves = [resultsFolderName+'DS_human_full.out']
    return trainSets, testSets, saves, predFNames, featureFolder


def loadDscriptData(resultsFolderName, spec_type = 'human'):
    featureFolder = os.path.join(path_root, 'dataset/preproc_data_DS/benchmark_feat/' + spec_type + '/')
    trainSets  = []
    testSets = []
    saves = []
    predFNames = []
    testSets = PPIPUtils.parseTSV(featureFolder + spec_type + '_test.tsv','string')
    testSets = [np.asarray(testSets)]
    predFNames = [resultsFolderName + 'pred_' + spec_type +'_DS.tsv']
    return trainSets, testSets, saves, predFNames, featureFolder


def loadLiADData(directory):
    currentDir = os.path.join(path_root, 'dataset/preproc_data_AD/benchmark_feat/')
    # currentDir = os.path.join(path_root, 'dataset/preproc_data/benchmark_feat/')

    trainSets = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Li_AD/li_AD_train_idx.tsv','int')
    testSets = PPIPUtils.parseTSV(currentDir+'PPI_Datasets/Li_AD/li_AD_test_idx.tsv','int')
    trainSets = [np.asarray(trainSets)]
    testSets = [np.asarray(testSets)]
    saves = [directory+'Li2020_AD.out']
    predFNames = [directory+'Li2020_AD_predict.tsv']
    
    return trainSets, testSets, saves,predFNames, currentDir+'PPI_Datasets/Li_AD/'


def augmentAll(trainSets,testSets):
    retSets = []
    for s in [trainSets,testSets]:
        if s is None:
            retSets.append(None)
        else:
            newSets = []
            for i in range(0,len(s)):
                newSets.append(augment(s[i]))
            retSets.append(newSets)
    return retSets
        
#create pair Y,X for every pair X,Y, and return only unique pairs
def augment(data):
    curData = np.asarray(data)
    #create mirror duplicate
    #index with None to replace dimension loss when indexing, and transpose to restore normal orientation
    curData2 = np.hstack((curData[None,:,1].T,curData[None,:,0].T,curData[None,:,2].T))
    #stack original and duplicate
    curData = np.vstack((curData,curData2))
        
    #remove duplicates, in case of (x,x) pairs, or pairs (x,y) and (y,x) being in the original data
    curData = np.unique(curData,axis=0)
    return curData

