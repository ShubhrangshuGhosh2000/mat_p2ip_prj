import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils.AACounter import AACounter

#1 epoch is more than enough to train a network this small with enough proteins
def OneHotEncoding(fastas, fileName, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],groupLen=1,sorting=False, flip=False,excludeSame=False,deviceType='cpu'):
    if groupings is not None:
        groupMap = {}
        idx = 0
        for item in groupings:
            for let in item:
                groupMap[let] = idx
            idx += 1
        numgroups = len(groupings)
        print(groupMap,numgroups)
    else:
        groupMap = None
        numgroups=20

    parsedData = AACounter(fastas, groupMap, groupLen, sorting=sorting,flip=flip,excludeSame=excludeSame,getRawValues=True,deviceType=deviceType)
    
    #number of unique groups, typically 20 amino acids, times length of our embeddings, typically 1, equals the corpus size
    corpusSize = numgroups*groupLen
    
    f = open(fileName,'w')
    #create Matrix
    for i in range(0,corpusSize):
        lst = [0] * corpusSize
        lst[i] = 1
        f.write(','.join(str(k) for k in lst)+'\n')
    f.write('\n')
    for item in parsedData[1:]:
        name = item[0]
        stVals = item[1].cpu().numpy()
        f.write(name+'\t'+','.join(str(k) for k in stVals) +'\n')
    f.close()

    return None
