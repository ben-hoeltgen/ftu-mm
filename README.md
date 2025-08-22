# Reconsidering Fairness Through Unawareness From the Perspective of Model Multiplicit

Codebase for the [paper of the same name](https://arxiv.org/abs/2505.16638) by Benedikt Höltgen and Nuria Oliver, presented and published at _ACM EAAMO 2025_.


## Repo guide
- `acs_single_feature.ipynb` and `main_results.ipynb` generate paper plots and tables based on results in the `results` folder
- the results can be replicated through `run_acs.py` and `run_almp.py`, given access to data
- `src` contains two files with helper functions, one for the notebooks and one for the 'run' files


## Datasets
- ACS Income and ACS Employment, available thorugh the `folktables` [package](https://github.com/socialfoundations/folktables) 
- Swiss Active Labour Market Policy (ALMP) Evaluation Dataset, access needs to be requested [here](https://www.swissubase.ch/en/catalogue/studies/13867/latest/datasets/1203/1953/overview) 


## Citing

```
@inproceedings{holtgen2025reconsidering,
  title={Reconsidering Fairness Through Unawareness From the Perspective of Model Multiplicity},
  author={H{\"o}ltgen, Benedikt and Oliver, Nuria},
  booktitle={The Fifth ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO'25)},
  year={2025}
}
```
