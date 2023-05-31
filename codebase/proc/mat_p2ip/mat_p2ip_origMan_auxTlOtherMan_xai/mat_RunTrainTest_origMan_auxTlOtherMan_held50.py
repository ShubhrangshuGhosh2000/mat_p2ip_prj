import sys, os
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


import numpy as np
from utils import PPIPUtils
import time


def writePredictions(fname,predictions,classes):
    f = open(fname,'w')
    for i in range(0,predictions.shape[0]):
        f.write(str(predictions[i])+'\t'+str(classes[i])+'\n')
    f.close()

def writeScore(predictions,classes, fOut, predictionsFName=None, thresholds=[0.01,0.03,0.05,0.1,0.25,0.5,1]):
    
    finalPredictions = np.hstack(predictions)
    finalClasses = np.hstack(classes)
    results = PPIPUtils.calcScores(finalClasses,finalPredictions,thresholds)
    
    
    lst = PPIPUtils.formatScores(results,'Total')
    for line in lst:
        fOut.write('\t'.join(str(s) for s in line) + '\n')
        print(line)
    
    fOut.write('\n')
    if predictionsFName is not None:
        writePredictions(predictionsFName,finalPredictions,finalClasses)



#modelClass - the class of the model to use
#testSets  -- lists of protein pairs to use in testings (p1, p2, class)
#featureFolder -- Folder containing dataset/features for classifier to pull features from
#hyperparams -- Any additional hyperparameters to pass into the model
def runTest_xai(modelClass, testSets, featureFolder, hyperParams = {}, predictionsFLst = None, startIdx=0, loads=None):
    print('Inside the runTest_xai() method - Start')
    if featureFolder[-1] != '/':
        featureFolder += '/'

    model = modelClass(hyperParams)
    model.loadFeatureData(featureFolder)
    
    for i in range(startIdx,len(testSets)):
        
        attrbn_df = None
        model.batchIdx = i
        model.loadModelFromFile(loads[i])
        
        attrbn_df = model.predictPairs_xai_humanBenchmark(testSets[i], predictionsFLst[i])
        
        attrbn_df_file_name = predictionsFLst[i].replace('.tsv', '_attrbn.tsv')
        attrbn_df.to_csv(attrbn_df_file_name, index=False)
    
    print('Inside the runTest_xai() method - End')



#modelClass - the class of the model to use
#outResultsName -- Name of File to write results to
#trainSets -- lists of protein pairs to use in trainings (p1, p2, class)
#testSets  -- lists of protein pairs to use in testings (p1, p2, class)
#featureFolder -- Folder containing dataset/features for classifier to pull features from
#hyperparams -- Any additional hyperparameters to pass into the model
#loadedModel -- Contains model already loaded with features, useful when combined with modelsLst argument to use different test sets on trained models
#modelsLst -- Optional argument containing list of already training models.  If provided, these models can be used in place of training (note:  argument is designed to take the list of models that this function returns so they can be used on a different test set)
def runTest(modelClass, outResultsName,trainSets,testSets,featureFolder,hyperParams = {},predictionsName =None,loadedModel= None,modelsLst = None,thresholds = [0.01,0.03,0.05,0.1,0.25,0.5,1],resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None, startIdx=0,loads=None):
    #record total time for computation
    t = time.time()
    
    if featureFolder[-1] != '/':
        featureFolder += '/'
    
    
    outResultDf = None
    outResultCsvFileName = outResultsName.replace('.txt', '.csv')
    if os.path.exists(outResultCsvFileName):
        outResultDf = pd.read_csv(outResultCsvFileName)
    

    #open file to write results to for each fold/split
    if resultsAppend and outResultsName:
        outResults = open(outResultsName,'a')
    elif outResultsName:
        outResults = open(outResultsName,'w')
    #keep list of predictions/classes per fold
    totalPredictions = []
    totalClasses = []
    trainedModelsLst = []

    for i in range(0,startIdx):
        totalPredictions.append([])
        totalClasses.append([])
        trainedModelsLst.append([])
    
    #create the model once, loading all the features and hyperparameters as necessary
    if loadedModel is not None:
        model = loadedModel
    else:
        model = modelClass(hyperParams)
        model.loadFeatureData(featureFolder)
    
    for i in range(startIdx,len(testSets)):
        model.batchIdx = i

        #create model, passing training data, testing data, and hyperparameters
        if modelsLst is None:
            model.batchIdx = i
            if loads is None or loads[i] is None:
                #run training and testing, get results
                model.train(trainSets[i])
                if saveModels is not None:
                    print('save')
                    model.saveModelToFile(saveModels[i])
            else:
                model.loadModelFromFile(loads[i])

            preds, classes = model.predictPairs(testSets[i])
            if keepModels:
                trainedModelsLst.append(model.getModel())
        else:
            #if we are given a model, skip the training and use it for testing
            preds, classes = model.predictPairs(testSets[i],modelsLst[i])
            if keepModels:
                trainedModelsLst.append(modelsLst[i])
        
        print('pred')
        #compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precition, and Max Precition @ k
        results = PPIPUtils.calcScores(classes,preds[:,1],thresholds)
        
        lst = PPIPUtils.formatScores(results,'Fold '+str(i))
        #write formatted results to file, and print result to command line
        if outResultsName:
            for line in lst:
                outResults.write('\t'.join(str(s) for s in line) + '\n')
                print(line)
            outResults.write('\n')
            
            outResults.close()
            outResults = open(outResultsName,'a')
            
            score_df = pd.DataFrame({'Fold': ['Fold '+str(i)]
                            , 'ACC': [results['ACC']], 'AUC': [results['AUC']], 'Prec': [results['Prec']], 'Recall': [results['Recall']]
                            , 'Thresholds': [results['Thresholds']], 'Max Precision': [results['Max Precision']]
                            , 'Avg Precision': [results['Avg Precision']]
                            })
            
            outResultDf = score_df if outResultDf is None else pd.concat([outResultDf, score_df], axis=0, sort=False)
            
            outResultDf.to_csv(outResultCsvFileName, index=False)
            
        else:
            for line in lst:
                print(line)
        print(time.time()-t)
        
        #append results to total results for overall scoring
        totalPredictions.append(preds[:,1])
        totalClasses.append(classes)

        if predictionsFLst is not None:
            writePredictions(predictionsFLst[i],totalPredictions[i],totalClasses[i])
            
    if not resultsAppend and predictionsName is not None and outResultsName: #not appending. calculate total results
        writeScore(totalPredictions,totalClasses,outResults,predictionsName,thresholds)
    
    
    if outResultsName:
        outResults.write('Time: '+str(time.time()-t))
        outResults.close()
    return (totalPredictions, totalClasses,model,trainedModelsLst)



def runTestLst_xai(modelClass, testSetsLst, featureFolder, hyperParams = {}, predictionsFLst = None, startIdx=0, loads=None):
    print('Inside the runTestLst_xai() method - Start')
    if featureFolder[-1] != '/':
        featureFolder += '/'

    model = modelClass(hyperParams)
    model.loadFeatureData(featureFolder)
    
    for i in range(startIdx,len(testSetsLst[0])):
        model.batchIdx = i
        model.loadModelFromFile(loads[i])
        
        for testIdx in range(0,len(testSetsLst)):
            attrbn_df = None
            attrbn_df = model.predictPairs_xai_humanBenchmark(testSetsLst[testIdx][i], predictionsFLst[testIdx][i])
            
            attrbn_df_file_name = predictionsFLst[testIdx][i].replace('.tsv', '_attrbn.tsv')
            attrbn_df.to_csv(attrbn_df_file_name, index=False)
    
    print('Inside the runTestLst_xai() method - End')



def runTestLst(modelClass, outResultsNameLst,trainSets,testSetsLst,featureFolder,hyperParams = {},predictionsNameLst=None,loadedModel= None,modelsLst = None,thresholds = [0.01,0.03,0.05,0.1,0.25,0.5,1],resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None,startIdx=0,loads=None):
    
    t = time.time()
    if featureFolder[-1] != '/':
        featureFolder += '/'
    
    outResultDfLst = []
    for i in range(0,len(testSetsLst)):
        indivOutResultCsvFileName = outResultsNameLst[i].replace('.txt', '.csv')
        outResultDfLst.insert(i, None)
        if os.path.exists(indivOutResultCsvFileName):
            outResultDfLst.insert(i, pd.read_csv(indivOutResultCsvFileName))
    
    totalPredictions = []
    totalClasses = []
    trainedModelsLst = []
    
    #open file to write results to for each fold/split
    outResults = []
    
    for i in range(0,len(testSetsLst)):
        if outResultsNameLst is not None:
            if resultsAppend:
                outResults.append(open(outResultsNameLst[i],'a'))
            else:
                outResults.append(open(outResultsNameLst[i],'w'))
        totalPredictions.append([])
        totalClasses.append([])
        trainedModelsLst.append([])
        
    for i in range(0,startIdx):
        for j in range(0,len(totalPredictions)):
            totalPredictions[j].append([])
            totalClasses[j].append([])
            trainedModelsLst[j].append([])
    
    #create the model once, loading all the features and hyperparameters as necessary
    if loadedModel is not None:
        model = loadedModel
    else:
        model = modelClass(hyperParams)
        model.loadFeatureData(featureFolder)
    
    for i in range(startIdx,len(testSetsLst[0])):
        model.batchIdx = i

        if modelsLst is None:
            model.batchIdx = i
            
            if loads is None or loads[i] is None:
                #run training and testing, get results
                model.train(trainSets[i])
                if saveModels is not None:
                    print('save')
                    model.saveModelToFile(saveModels[i])
            else:
                print('load')
                model.loadModelFromFile(loads[i])
                
            if keepModels:
                trainedModelsLst.append(model.getModel())    
            
        else:
            #if we are given a model, skip the training and use it for testing
            if keepModels:
                trainedModelsLst.append(modelsLst[i])
        
        for testIdx in range(0,len(testSetsLst)):
            if modelsLst is None:
                preds, classes = model.predictPairs(testSetsLst[testIdx][i])
            else:
                preds, classes = model.predictPairs(testSetsLst[testIdx][i],modelsLst[i])
            
            #compute result metrics, such as Max Accuracy, Precision/Recall @ Max Accuracy, Average Precition, and Max Precition @ k
            results = PPIPUtils.calcScores(classes,preds[:,1],thresholds)
            #format the scoring results, with line for title
            lst = PPIPUtils.formatScores(results,'Fold '+str(i))
            #write formatted results to file, and print result to command line
            if outResultsNameLst is not None:
                for line in lst:
                    outResults[testIdx].write('\t'.join(str(s) for s in line) + '\n')
                    print(line)
                outResults[testIdx].write('\n')
                
                outResults[testIdx].close()
                outResults[testIdx] = open(outResultsNameLst[testIdx],'a')
                
                score_df = pd.DataFrame({'Fold': ['Fold '+str(i)]
                                , 'ACC': [results['ACC']], 'AUC': [results['AUC']], 'Prec': [results['Prec']], 'Recall': [results['Recall']]
                                , 'Thresholds': [results['Thresholds']], 'Max Precision': [results['Max Precision']]
                                , 'Avg Precision': [results['Avg Precision']]
                                })
                
                outResultDfLst[testIdx] = score_df if outResultDfLst[testIdx] is None else pd.concat([outResultDfLst[testIdx], score_df], axis=0, sort=False)
                
                outResultDfLst[testIdx].to_csv(outResultsNameLst[testIdx].replace('.txt', '.csv'), index=False)
                
            else:
                for line in lst:
                    print(line)
            print(time.time()-t)
            
            totalPredictions[testIdx].append(preds[:,1])
            totalClasses[testIdx].append(classes)
            if predictionsFLst is not None:
                writePredictions(predictionsFLst[testIdx][i],totalPredictions[testIdx][i],totalClasses[testIdx][i])
    if outResultsNameLst is not None:
        for testIdx in range(0,len(testSetsLst)):
            if not resultsAppend: 
                writeScore(totalPredictions[testIdx],totalClasses[testIdx],outResults[testIdx],(predictionsNameLst[testIdx] if predictionsNameLst is not None else None),thresholds)
            
            #output the total time to run this algorithm
            outResults[testIdx].write('Time: '+str(time.time()-t))
            outResults[testIdx].close()
    return (totalPredictions, totalClasses,model,trainedModelsLst)


def calcOverallScore_Pos50(outResultsName):
    outResultCsvFileName = outResultsName.replace('.txt', '.csv')
    outResultDf = pd.read_csv(outResultCsvFileName)
    
    avg_ACC = outResultDf['ACC'].mean()
    avg_AUC = outResultDf['AUC'].mean()
    
    outResultDf['avg_ACC'] = [avg_ACC] + [''] * (outResultDf.shape[0] - 1)
    outResultDf['avg_AUC'] = [avg_AUC] + [''] * (outResultDf.shape[0] - 1)
    
    outResultDf.to_csv(outResultCsvFileName, index=False)
    


def calcOverallScore_Pos20(outResultsNameLst):
    for outResultsName in outResultsNameLst:
        
        outResultCsvFileName = outResultsName.replace('.txt', '.csv')
        outResultDf = pd.read_csv(outResultCsvFileName)
        
        con_Prec_lst = []
        max_p_arr = outResultDf['Max Precision'].to_numpy()
        
        for indiv_Max_Precision_str in max_p_arr:
            
            indiv_Max_Precision_str = indiv_Max_Precision_str.replace('[', '').replace(']', '')  
            indiv_Max_Precision_lst = [float(prec) for prec in indiv_Max_Precision_str.split(',')]
            indiv_Prec_val = indiv_Max_Precision_lst[1]  
            con_Prec_lst.append(indiv_Prec_val)
        avg_Prec = np.asarray(con_Prec_lst).mean()
        
        con_avg_p_lst = []
        avg_p_arr = outResultDf['Avg Precision'].to_numpy()
        
        for indiv_Avg_P_str in avg_p_arr:
            
            indiv_Avg_P_str = indiv_Avg_P_str.replace('[', '').replace(']', '')  
            indiv_Avg_P_lst = [float(prec) for prec in indiv_Avg_P_str.split(',')]
            indiv_Avg_P_val = indiv_Avg_P_lst[-1]  
            con_avg_p_lst.append(indiv_Avg_P_val)
        avg_Avg_P = np.asarray(con_avg_p_lst).mean()
        
        outResultDf['avg_Prec'] = [avg_Prec] + [''] * (outResultDf.shape[0] - 1)
        outResultDf['avg_Avg_P'] = [avg_Avg_P] + [''] * (outResultDf.shape[0] - 1)
        
        outResultDf.to_csv(outResultCsvFileName, index=False)
    
