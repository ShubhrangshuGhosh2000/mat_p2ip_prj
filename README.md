# MaTPIP
The codebase is associated with the paper "MaTPIP: a deep-learning architecture with eXplainable AI for sequence-driven, feature mixed protein-protein interaction prediction"

 * We have used conda environment with Python 3.8.1 for execution. The conda environment is created using py381_gpu_param.yml file.

 * To run a complete code-flow in any of the modules in the 'proc' folder, please check the files with name pattern  xxx_RunTests_yyy.py. 

 * Some abbreviations used in the codebase are:
AD: Alzheimer Disease; DS: Different Species (Mosuse, Fly, Worm, Yeast, Ecoli).

 * 'origMan' in the codebase implies 2-Dimensional Manually curated Features. Similary 'auxTl' and 'OtherMan' correspond to 1-Dimensional pre-trained Protein Language Model(PLM)-based Features and 1-Dimensional Manually curated Features respectively.

 * There are redundancies in the codebase (specially in 'mat_p2ip' folder) as we used to run different sets of experiments concurrently on a single machine.

