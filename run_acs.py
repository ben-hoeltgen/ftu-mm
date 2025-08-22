import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path

from src.utils import *



### SETTINGS

TASK = 'income' # 'income' or 'employment'
PROT_ATT = 'sex' # 'race' or 'sex'
MODEL_CLASS = 'LR' # 'LR' or 'GBM_opt'

states = ['ALL']
income_thresholds = np.arange(5, 6) * 1e4 # np.arange(1, 8) * 1e4 is range used in Ding paper

N_SPLIT_SEEDS = 10



### RUN

if TASK == 'employment':
    income_thresholds = [0]

prot_feature = 'RAC1P' if PROT_ATT == 'race' else 'SEX' if PROT_ATT == 'sex' else ValueError('PROT_ATT must be race or sex')	



## create empty arrays to fill
accuracy_array = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
optim_acc_array = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
optim_acc_array_train = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
auroc_array = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
auroc_PA_array = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
disp_imp_array = np.empty((len(states), len(income_thresholds), 2, N_SPLIT_SEEDS))
base_rate_array = np.empty((len(states), len(income_thresholds), 3))


## create paths to save results in
result_folder = f"results/acs-{TASK}-{PROT_ATT}-{MODEL_CLASS}"
Path(result_folder).mkdir(exist_ok=True)


## loop over data generation (for robustness)
for state_idx in range(len(states)):
    
    print('    --------   now working with', states[state_idx], 'data   --------     ')
    for it_idx in range(len(income_thresholds)):

        for PA_included in [0, 1]: # run unaware then aware version

            features, label, group = create_acs_dataset(TASK, states[state_idx], income_thresholds[it_idx], PA_included, 
                                                    protected_att=prot_feature, only_features=None)
            
            base_rate_array[state_idx][it_idx][PA_included] = label[group == (PA_included+1)].mean()
            base_rate_array[state_idx][it_idx][2] = label.mean()

            if PA_included == 0:
                indices = np.arange(len(features)) # to make sure both aware and unaware use the same split, for direct comparison

            # loop over splits (for robustness)
            for split_seed in tqdm(range(N_SPLIT_SEEDS)):

                X_train, X_test, y_train, y_test, group_train, group_test, idx_train, idx_test = train_test_split(features, label, group, indices,
                                                                                                                  test_size=0.2, random_state=split_seed)

                # fit model
                model = fit_model(X_train, y_train, MODEL_CLASS)

                # test set prediction
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:,1]

                # evaluate accuracy etc
                accuracy_array[state_idx][it_idx][PA_included][split_seed] = accuracy_score(y_test, y_pred)
                auroc_array[state_idx][it_idx][PA_included][split_seed] = roc_auc_score(y_test, y_pred)
                auroc_PA_array[state_idx][it_idx][PA_included][split_seed] = roc_auc_score(group_test==1, y_pred)

                # disparate impact
                disp_imp_array[state_idx][it_idx][PA_included][split_seed] = y_pred[group_test==1].mean() - y_pred[group_test==2].mean()
                
                ### accuracy of Bayes-optimal classifier
                optim_acc_array[state_idx][it_idx][PA_included][split_seed] = optimal_acc(X_test, y_test)
                optim_acc_array_train[state_idx][it_idx][PA_included][split_seed] = optimal_acc(X_train, y_train)

    
    print('proportion group 1:', (group_test == 1).mean())

    ### dump files: one row per seed
    for i in [0,1]:    
        df_accuracy = pd.DataFrame(data=accuracy_array[state_idx][:,i].transpose(), columns=income_thresholds)
        df_accuracy.to_csv(result_folder + f'/accuracies_{states[state_idx]}_{i}.csv')

        df_auroc = pd.DataFrame(data=auroc_array[state_idx][:,i].transpose(), columns=income_thresholds)
        df_auroc.to_csv(result_folder + f'/aurocs_{states[state_idx]}_{i}.csv')

        df_optim_acc = pd.DataFrame(data=optim_acc_array[state_idx][:,i].transpose(), columns=income_thresholds)
        df_optim_acc.to_csv(result_folder + f'/optim_accs_{states[state_idx]}_{i}.csv')

        df_optim_acc_train = pd.DataFrame(data=optim_acc_array_train[state_idx][:,i].transpose(), columns=income_thresholds)
        df_optim_acc_train.to_csv(result_folder + f'/optim_accs_train_{states[state_idx]}_{i}.csv')

        df_auroc_PA = pd.DataFrame(data=auroc_PA_array[state_idx][:,i].transpose(), columns=income_thresholds)
        df_auroc_PA.to_csv(result_folder + f'/aurocs_PA_{states[state_idx]}_{i}.csv')
            
        df_ineq = pd.DataFrame(data=disp_imp_array[state_idx][:,i].transpose(), columns=income_thresholds)
        df_ineq.to_csv(result_folder + f'/inequalities_{states[state_idx]}_{i}.csv')
            
    df_base_rate = pd.DataFrame(data=base_rate_array[state_idx].transpose(), columns=income_thresholds)
    df_base_rate.to_csv(result_folder + f'/base_rates_{states[state_idx]}.csv')