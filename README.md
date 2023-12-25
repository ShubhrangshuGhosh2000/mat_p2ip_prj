# MaTPIP
This repository contains the code for the paper [MaTPIP: a deep-learning architecture with eXplainable AI for sequence-driven, feature mixed protein-protein interaction prediction](https://doi.org/10.1016/j.cmpb.2023.107955) by Shubhrangshu Ghosh and Pralay Mitra.

### Abstract

*Background and Objective:* Protein-protein interaction (PPI) is a vital process in all living cells, controlling essential cell functions such as cell cycle regulation, signal transduction, and metabolic processes with broad applications that include antibody therapeutics, vaccines, and drug discovery. The problem of sequence-based PPI prediction has been a long-standing issue in computational biology.

*Methods:* We introduce MaTPIP, a cutting-edge deep-learning framework for predicting PPI. MaTPIP stands out due to its innovative design, fusing pre-trained Protein Language Model (PLM)-based features with manually curated protein sequence attributes, emphasizing the part-whole relationship by incorporating two-dimensional granular part (amino-acid) level features and one-dimensional whole-level (protein) features. What sets MaTPIP apart is its ability to integrate these features across three different input terminals seamlessly. MatPIP also includes a distinctive configuration of Convolutional Neural Network (CNN) with Transformer components for concurrent utilization of CNN and sequential characteristics in each iteration and a one-dimensional to twodimensional converter followed by a unified embedding. The statistical significance of this classifier is validated using McNemar’s test.

*Results:* MaTPIP outperformed the existing methods on both the Human PPI benchmark and cross-species PPI testing datasets, demonstrating its immense generalization capability for PPI prediction. We used seven diverse datasets with varying PPI target class distributions. Notably, within the novel PPI scenario, the most challenging category for Human PPI Benchmark, MaTPIP improves the existing state-of-the-art score from 74.1% to 78.6% (measured in Area under ROC Curve), from 23.2% to 32.8% (in average precision) and from 4.9% to 9.5% (in precision at 3% recall) for 50%, 10% and 0.3% target class distributions, respectively. In cross-species PPI evaluation, hybrid MaTPIP establishes a new benchmark  score (measured in Area Under precision-recall curve) of 81.1% from the previous 60.9% for Mouse, 80.9% from 56.2% for Fly, 78.1% from 55.9% for Worm, 59.9% from 41.7% for Yeast, and 66.2% from 58.8% for E.coli. Our eXplainable AI-based assessment reveals an average contribution of different feature families per prediction on these datasets.

*Conclusions:* MaTPIP mixes manually curated features with the feature extracted from the pre-trained PLM to predict sequence-based protein-protein association. Furthermore, MaTPIP demonstrates strong generalization capabilities for cross-species PPI predictions.

## Instructions for the codebase

 * We have used conda environment with Python 3.8.1 for execution. The conda environment is created using *py381_gpu_param.yml* file.

 * To run a complete code-flow in any of the modules in the 'proc' folder, please check the files with name pattern *xxx_RunTests_yyy.py*. 

 * Some abbreviations used in the codebase are:
AD: Alzheimer Disease; DS: Different Species (Mosuse, Fly, Worm, Yeast, Ecoli).

 * 'origMan' in the codebase implies 2-Dimensional Manually curated Features. Similary 'auxTl' and 'OtherMan' correspond to 1-Dimensional pre-trained Protein Language Model (PLM)-based Features and 1-Dimensional Manually curated Features respectively.

 * There are redundancies in the codebase (specially in 'mat_p2ip' folder) as we used to run different sets of experiments concurrently on a single machine.

## Citation
```
@article{ghosh2023matpip,
  title={MaTPIP: A deep-learning architecture with eXplainable AI for sequence-driven, feature mixed protein-protein interaction prediction},
  author={Ghosh, Shubhrangshu and Mitra, Pralay},
  journal={Computer Methods and Programs in Biomedicine},
  pages={107955},
  year={2023},
  publisher={Elsevier}
}
```
