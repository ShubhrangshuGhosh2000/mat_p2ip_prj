import sys

from pathlib import Path

path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from AACounter import AACounter


def AAC(fastas, groupings = None, groupLen=1, normType='100',deviceType='cpu'):
	if groupings is not None:
		groupMap = {}
		idx = 0
		for item in groupings:
			for let in item:
				groupMap[let] = idx
			idx += 1
	else:
		groupMap = None
		
	return AACounter(fastas,groupMap,groupLen,normType=normType,deviceType=deviceType)
