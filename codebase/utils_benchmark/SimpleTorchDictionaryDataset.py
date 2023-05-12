import torch
from torch.utils import data

#uses a matrix of proteins, and indexes with a matrix of pairs in the form of (x0Idx, x1Idx, classVal)
class SimpleTorchDictionaryDataset(data.Dataset):
    def __init__(self,featureData,pairLst,classData=None,full_gpu=False,deviceType='cpu',createNewTensor=False):
        if createNewTensor:
            self.data = torch.tensor(featureData).float()
        else:
            self.data=featureData
        
        self.pairLst =pairLst#torch.tensor(pairLst).long()
        
        self.noClasses=False
        
        #network runner predict checks for a -1 as the first value to mean no class data provided
        #assigning a list of -1's as class allows us to:
            #avoid an extra if statement in the dataset function
            #ensure that the same amount of data is returned (data,class) in each call even when predicting with no known class
            #ensure the torch collate function will work even when no class data exists
        #please don't use a class value of -1 if intending to pass classes to the predict function (eval)
        if classData is None:
            self.noClasses=True
            self.classData = torch.ones(self.pairLst.shape[0])*-1
        else:
            self.classData = torch.tensor(classData)
        self.classData = self.classData.long()
        
        self.full_gpu = full_gpu #if true, push everything to gpu
        self.deviceType = deviceType


    def __getitem__(self,index):
        y = self.classData[index]
        #individually indexing is faster?
        x0 = self.data[self.pairLst[index][0]]
        x1 = self.data[self.pairLst[index][1]]
        x0 = x0.unsqueeze(0)
        x1 = x1.unsqueeze(0)
        x0 = x0.float()
        x1 = x1.float()
        return (x0,x1,y)


    def __len__(self):
        return self.classData.shape[0]


    def activate(self):        
        if self.full_gpu: #push everything to gpu
            # self.data = self.data.cuda()
            self.data = self.data.to(torch.device(self.deviceType))
            #self.pairLst = self.pairLst.cuda()
            # self.classData = self.classData.cuda()
            self.classData = self.classData.to(torch.device(self.deviceType))


    def deactivate(self):
        self.data = self.data.cpu()
        #self.pairLst = self.pairLst.cpu()
        self.classData = self.classData.cpu()
