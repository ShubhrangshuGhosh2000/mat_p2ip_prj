import sys
import pandas as pd

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)


import numpy as np
# import PPIPUtils
from utils import PPIPUtils
import time


def writePredictions(fname,predictions,classes):
    f = open(fname,'w')
    for i in range(0,predictions.shape[0]):
        f.write(str(predictions[i])+'\t'+str(classes[i])+'\n')
    f.close()

def writeScore(predictions,classes, fOut, predictionsFName=None, thresholds=[0.01,0.03,0.05,0.1,0.25,0.5,1]):
    #concate results from each fold, and get total scoring metrics
    finalPredictions = np.hstack(predictions)
    finalClasses = np.hstack(classes)
    results = PPIPUtils.calcScores(finalClasses,finalPredictions,thresholds)
    
    #format total metrics, and write them to a file
    lst = PPIPUtils.formatScores(results,'Total')
    for line in lst:
        fOut.write('\t'.join(str(s) for s in line) + '\n')
        print(line)
    
    fOut.write('\n')
    
    if predictionsFName is not None:
        writePredictions(predictionsFName,finalPredictions,finalClasses)



# SPECIALLY WRITTEN FOR Different Species (DS): full human data training
def runTrainOnly_DS(modelClass, trainSets, featureFolder, hyperParams, saveModels, spec_type):
    print('Inside the runTrainOnly_DS() method - Start')
    #record total time for computation
    t = time.time()
    
    if featureFolder[-1] != '/':
        featureFolder += '/'

    model = modelClass(hyperParams)
    model.loadFeatureData_DS(featureFolder, spec_type)
    model.train(trainSets[0])
    if saveModels is not None:
        print('save')
        model.saveModelToFile(saveModels[0])
    print('Time: '+str(time.time()-t))
    print('Inside the runTrainOnly_DS() method - End')
    return model


# SPECIALLY WRITTEN FOR Different Species (DS).
#modelClass - the class of the model to use
#outResultsName -- Name of File to write results to
#trainSets -- lists of protein pairs to use in trainings (p1, p2, class)
#testSets  -- lists of protein pairs to use in testings (p1, p2, class)
#featureFolder -- Folder containing dataset/features for classifier to pull features from
#hyperparams -- Any additional hyperparameters to pass into the model
#loadedModel -- Contains model already loaded with features, useful when combined with modelsLst argument to use different test sets on trained models
#modelsLst -- Optional argument containing list of already training models.  If provided, these models can be used in place of training (note:  argument is designed to take the list of models that this function returns so they can be used on a different test set)
def runTest_DS(modelClass, outResultsName,trainSets,testSets,featureFolder,hyperParams = {},predictionsName =None,loadedModel= None,modelsLst = None,resultsAppend=False,keepModels=False,saveModels=None,predictionsFLst = None, startIdx=0,loads=None,spec_type=None):
    print('Inside the runTest_DS() method - Start')
    #record total time for computation
    t = time.time()
    
    if featureFolder[-1] != '/':
        featureFolder += '/'
    # ############### EXTRA CODE ADDED -START #####################
    # check whether outResults CSV exists; if it exists, then load it
    outResultDf = None
    outResultCsvFileName = outResultsName.replace('.txt', '.csv')
    # if os.path.exists(outResultCsvFileName):  # For Different Species, no need to aapend
    #     outResultDf = pd.read_csv(outResultCsvFileName)
    # ############### EXTRA CODE ADDED -END #####################

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
        model.loadFeatureData_DS(featureFolder, spec_type)
    
    for i in range(startIdx,len(testSets)):
        # for i in range(2,len(testSets)):  # ############ TEMP CODE
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
                #model.setScaleFeatures(trainSets[i])

            preds, classes = model.predictPairs(testSets[i])
            if keepModels:
                trainedModelsLst.append(model.getModel())
        else:
            #if we are given a model, skip the training and use it for testing
            preds, classes = model.predictPairs(testSets[i],modelsLst[i])
            if keepModels:
                trainedModelsLst.append(modelsLst[i])
        
        print('prediction result processing')
        # compute result metrics, such as AUPR, Precision, Recall, AUROC
        results = PPIPUtils.calcScores_DS(classes,preds[:,1])
        #format the scoring results, with line for title
        lst = PPIPUtils.formatScores_DS(results,'Species: '+spec_type)
        #write formatted results to file, and print result to command line
        if outResultsName:
            for line in lst:
                outResults.write('\t'.join(str(s) for s in line) + '\n')
                print(line)
            outResults.write('\n')
            # ############### EXTRA CODE ADDED -START #####################
            outResults.close()
            outResults = open(outResultsName,'a')
            # create score_df
            score_df = pd.DataFrame({'Species': [spec_type]
                            , 'AUPR': [results['AUPR']], 'Precision': [results['Precision']]
                            , 'Recall': [results['Recall']], 'AUROC': [results['AUROC']]
                            })
            # store the score_df into outResultDf
            outResultDf = score_df if outResultDf is None else pd.concat([outResultDf, score_df], axis=0, sort=False)
            # save outResultDf as CSV
            outResultDf.to_csv(outResultCsvFileName, index=False)
            # ############### EXTRA CODE ADDED -END #####################
        else:
            for line in lst:
                print(line)
        print(time.time()-t)

        #append results to total results for overall scoring
        totalPredictions.append(preds[:,1])
        totalClasses.append(classes)

        if predictionsFLst is not None:
            writePredictions(predictionsFLst[i],totalPredictions[i],totalClasses[i])
    # end of for loop: for i in range(startIdx,len(testSets)):

    # if not resultsAppend and predictionsName is not None and outResultsName: #not appending. calculate total results
    #     writeScore(totalPredictions,totalClasses,outResults,predictionsName,thresholds)
    
    #output the total time to run this algorithm
    if outResultsName:
        outResults.write('Time: '+str(time.time()-t))
        outResults.close()
    print('Inside the runTest_DS() method - End')
    return (totalPredictions, totalClasses, model, trainedModelsLst)


# SPECIALLY WRITTEN FOR Different Species (DS): feature attribution purpose .
#modelClass - the class of the model to use
#testSets  -- lists of protein pairs to use in testings (p1, p2, class)
#featureFolder -- Folder containing dataset/features for classifier to pull features from
#hyperparams -- Any additional hyperparameters to pass into the model
#loadedModel -- Contains model already loaded with features, useful when combined with modelsLst argument to use different test sets on trained models
#modelsLst -- Optional argument containing list of already training models.  If provided, these models can be used in place of training (note:  argument is designed to take the list of models that this function returns so they can be used on a different test set)
def runTest_DS_xai(modelClass,testSets,featureFolder,hyperParams = {},startIdx=0,loads=None,spec_type=None,resultsFolderName=None):
    print('Inside the runTest_DS_xai() method - Start')
    if featureFolder[-1] != '/':
        featureFolder += '/'

    model = modelClass(hyperParams)
    model.loadFeatureData_DS(featureFolder, spec_type)
    
    for i in range(startIdx,len(testSets)):
        model.loadModelFromFile(loads[i])
        model.predictPairs_xai_DS(testSets[i],resultsFolderName)
    # end of for loop: for i in range(startIdx,len(testSets)):
    print('Inside the runTest_DS_xai() method - End')

