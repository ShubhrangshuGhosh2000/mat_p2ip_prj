# MaTPPIp
The codebase is associated with the paper "MaTPPIp : a sequence driven, feature mixed protein-protein interaction prediction architecture with eXplainable AI based attribute assessment"

 1. We have used conda environment with Python 3.8.1 for execution. The conda environment is created using py381_gpu_param.yml file.

 2. To run a complete code-flow in any of the modules in the 'proc' folder, please check the files with name pattern  xxx_RunTests_yyy.py. 

 3. Some abbreviations used in the codebase are:
AD: Alzheimer Disease; DS: Different Species (Mosuse, Fly, Worm, Yeast, Ecoli).

 4. 'origMan' in the codebase implies 2-Dimensional Manually curated Features. Similary 'auxTl' and 'OtherMan' correspond to 1-Dimensional Transfer Learning based Features and 1-Dimensional Manually curated Features respectively.

 5. There are redundancies in the codebase (specially in 'mtf_p2ip' folder) as we used to run different sets of experiments concurrently on a single machine.



### (\*) remove all the comments by using regular expression:  #.* with blank (no space) at the end, after taking a backup of codebase with all the comments. 
