import numpy as np
from sklearn import metrics


def calcScores(classData,preds,thresholds):
	predictions = np.vstack((preds,classData)).T
	(prec,recall,acc,tnr) = calcPrecisionRecallLsts(predictions)
	scores = {'Avg Precision':[],'Max Precision':[],'Thresholds':thresholds,'AUC':0,'ACC':0,'Prec':0,'Recall':0}
	for item in thresholds:
		auprc = calcAUPRCLsts([prec],[recall],maxRecall=item)[0]
		precision = np.max(prec[recall>=item])
		scores['Avg Precision'].append(auprc)
		scores['Max Precision'].append(precision)

	auc = calcAUROC(classData,preds)
	maxAcc = np.max(acc)
	maxAccIdx = np.argmax(acc>=maxAcc)
	scores['AUC'] = auc
	scores['ACC'] = maxAcc
	scores['Prec'] = prec[maxAccIdx]
	scores['Recall'] = recall[maxAccIdx]
	return scores


def calcPrecisionRecallLsts(lst):
	#finalR = []
	#finalP = []
	#finalR.append([])
	#finalP.append([])
	lst = np.asarray(lst)
	ind = np.argsort(-lst[:,0])
	lst = lst[ind,:]
	#get total true and cumulative sum of true
	totalPositive = np.sum(lst[:,1])
	totalNegative = lst.shape[0]-totalPositive
	
	finalR = np.cumsum(lst[:,1])
	FP = np.arange(1,finalR.shape[0]+1)-finalR
	
	#create precision array (recall counts / total counts)
	finalP = finalR/np.arange(1,lst.shape[0]+1)
	
	#find ties
	x = np.arange(finalR.shape[0]-1)
	ties = list(np.where(lst[x,0] == lst[x+1,0])[0])
	for idx in range(len(ties)-1,-1,-1):
		finalR[ties[idx]] = finalR[ties[idx]+1]
		finalP[ties[idx]] = finalP[ties[idx]+1]
		FP[ties[idx]] = FP[ties[idx]+1]

	TN = totalNegative - FP
	ACC = (TN + finalR)/finalR.shape[0]
	TNR = TN/totalNegative
	
	#scale recall from 0 to 1
	finalR = finalR / totalPositive	
	
	return (finalP,finalR,ACC,TNR)


def calcAUPRCLsts(finalP,finalR,maxRecall=0.2):
	scores = []
	for i in range(0,len(finalR)):
		#do cutoffs
		cutoff = np.argmax(finalR[i]>maxRecall)
		if cutoff == 0:
			if maxRecall >= np.max(finalR[i]): #100% recall, grab all data
				cutoff = finalR[i].shape[0]-1
				
		cutoff+=1		
		pData = finalP[i][0:cutoff]
		rData = finalR[i][0:cutoff]
		
		
		
		#create binary vector of true values
		rData2 = np.hstack((np.zeros(1),rData[:-1]))
		rData = rData-rData2
		
		#calculate average precision at each true value
		x1 = np.sum(rData * pData)/np.sum(rData)
		
		scores.append(x1)
	return scores


def calcAUROC(classData,preds):
	fpr, tpr, thresholds = metrics.roc_curve(classData, preds)
	return metrics.auc(fpr, tpr)


