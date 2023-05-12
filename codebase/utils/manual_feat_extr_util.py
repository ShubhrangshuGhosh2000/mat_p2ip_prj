import os
import sys

from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

# see https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-the-currently-running-scrip
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def getAALookup():
    AA = 'ARNDCQEGHILKMFPSTWYV'
    AADict= {}
    for item in AA:
        AADict[item] = len(AADict)
    return AADict


def loadAAData(aaIDs):
    AADict = getAALookup()

    #get aa index data given aaIDs
    aaIdx = open(os.path.join(__location__, 'AAidx.txt'))
    aaData = []
    for line in aaIdx:
        aaData.append(line.strip())

    myDict = {}
    for idx in range(1,len(aaData)):
        data = aaData[idx].strip().split('\t')
        myDict[data[0]] = data[1:]  
    AAProperty = []
    for item in aaIDs:
        AAProperty.append([float(j) for j in myDict[item]])

    return AADict, aaIDs, AAProperty


def loadPairwiseAAData(aaIDs,AADict=None):
    AADict = getAALookup()
    aaIDs = aaIDs
    AAProperty = []
    for item in aaIDs:
        if item == 'Grantham':
            f = open(os.path.join(__location__, 'Grantham.txt'))
        elif item == 'Schneider-Wrede':
            f = open(os.path.join(__location__, 'Schneider-Wrede.txt'))
        
        data = []
        colHeaders = []
        rowHeaders = []
        for line in f:
            line = line.strip().split()
            if len(line) > 0:
                if len(colHeaders) == 0:
                    colHeaders = line
                else:
                    rowHeaders.append(line[0])
                    for i in range(1,len(line)):
                        line[i] = float(line[i])
                    data.append(line[1:])
        f.close()
        AAProperty.append(data)
    return AADict, aaIDs, AAProperty


#local descriptor 10
#splits each protein sequence into 10 parts prior to computing sequence-based values
def LDEncode10(fastas,uniqueString='_+_'):
    newFastas = []
    for item in fastas:
        name = item[0]
        st = item[1]
        intervals = [0,len(st)//4,len(st)//4*2,len(st)//4*3,len(st)]
        mappings= []
        idx = 0
        for k in range(1,5):
            for i in range(0,5-k):
                newName=name+uniqueString+str(idx)
                if i == 0 and k == 4:
                    #compute middle 75%
                    newString = st[len(st)//8:len(st)//8*7]
                else:
                    newString = st[intervals[i]:intervals[i+k]]
                newFastas.append([newName,newString])
                idx += 1
    return (newFastas,10)


def STDecode(values,parts=10,uniqueString='_+_'):
    #final data list
    valLst = []
    #remap values to original proteins
    valDict = {}
    nameOrder= []
    for line in values:
        if len(valLst) == 0:
            #header
            lst = [line[0]]
            for j in range(0,parts):
                for i in range(1,len(line)):
                    lst.append(line[i]+'_'+str(j))
            valLst.append(lst) 
            continue
        line[0] = line[0].split(uniqueString)
        realName = line[0][0]
        idx = int(line[0][1])
        if realName not in valDict:
            valDict[realName] = [None]*parts
            nameOrder.append(realName)
        valDict[realName][idx] = line[1:]
        
    #error checking, and return all data
    for item in nameOrder:
        a = [item] #name
        for i in range(0,parts):
            if valDict[item][i] is None:
                print('Error, Missing Data Decode',item,i)
                exit(42)
            a.extend(valDict[item][i])
        valLst.append(a)
    return valLst
