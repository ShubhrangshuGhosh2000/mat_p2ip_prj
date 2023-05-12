import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils_benchmark.AACounter import AACounter

def ConjointTriad(fastas, deviceType='cpu'):
	groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C']
	groupLen=3
	groupMap = {}
	idx = 0
	for item in groupings:
		for let in item:
			groupMap[let] = idx
		idx += 1
	return AACounter(fastas,groupMap,groupLen,normType='CTD',deviceType=deviceType)
