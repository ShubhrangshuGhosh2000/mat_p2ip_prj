import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils.AACounter import AACounter

#by default, use CTD groupings
#Based on paper Prediction of protein folding class using global description of amino acid sequence  INNA DUBCHAK, ILYA MUCHNIKt, STEPHEN R. HOLBROOK, AND SUNG-HOU KIM
#Based on paper Prediction of protein allergenicity using local description of amino acid sequence, by Joo Chuan Tong, Martti T. Tammi,
#First Paper listed multiplied calculed percentaged by 100 before using them, but I'm leaving them as floats between 0-1 as that matches most other data types better.
def CTD_Composition(fastas, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'],deviceType='cpu'):
    if groupings is not None:
        groupMap = {}
        idx = 0
        for item in groupings:
            for let in item:
                groupMap[let] = idx
            idx += 1
    else:
        groupMap = None
        
    return AACounter(fastas,groupMap,normType='100',deviceType=deviceType)
