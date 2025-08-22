import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path

from src.utils import *



### SETTINGS

PROT_ATT = 'citizen' # 'gender' or 'citizen'
MODEL_CLASS = 'LR' # 'LR' or 'GBM_opt'

N_SPLIT_SEEDS = 10



### RUN

if PROT_ATT == 'gender':
    prot_feature = 'female' 
elif PROT_ATT == 'citizen':
    prot_feature = 'swiss'
else:
    ValueError('PROT_ATT must be "gender" or "citizen"')	



## create empty arrays to fill
accuracy_array = np.empty((2, N_SPLIT_SEEDS))
optim_acc_array = np.empty((2, N_SPLIT_SEEDS))
optim_acc_array_train = np.empty((2, N_SPLIT_SEEDS))
auroc_array = np.empty((2, N_SPLIT_SEEDS))
pred_inequality_array = np.empty((2, N_SPLIT_SEEDS))
disp_imp_array = np.empty((2, N_SPLIT_SEEDS))
group_c_rate_array = np.empty((2, 2, N_SPLIT_SEEDS))


## create paths to save results in
result_folder = f"results/almp-{PROT_ATT}-{MODEL_CLASS}"
Path(result_folder).mkdir(exist_ok=True)


## loop over data generation (for robustness)
for PA_included in [0, 1]:

    features, label, group = create_almp_dataset(protected_att=prot_feature, include_protected=PA_included)
    
    base_rate_diff = label[group == 1].mean() - label[group == 2].mean() # pos if group 1 has higher base rate

    # loop over splits (for robustness)
    for split_seed in tqdm(range(N_SPLIT_SEEDS)):

        X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(features, label, group, 
                                                                            test_size=0.2, random_state=split_seed)

        # fit model
        model = fit_model(X_train, y_train, MODEL_CLASS)

        # test set prediction
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]

        # evaluate accuracy and auroc
        accuracy_array[PA_included][split_seed] = accuracy_score(y_test, y_pred)
        auroc_array[PA_included][split_seed] = roc_auc_score(y_test, y_pred_proba)

        # avg predictions
        avg_pred_1 = y_pred_proba[group_test==1].mean()
        avg_pred_2 = y_pred_proba[group_test==2].mean()

        # predictive inequality and disparate impact
        pred_inequality_array[PA_included][split_seed] = avg_pred_1 - avg_pred_2
        disp_imp_array[PA_included][split_seed] = y_pred[group_test==1].mean() - y_pred[group_test==2].mean()

        ### accuracy of Bayes-optimal classifier
        optim_acc_array[PA_included][split_seed] = optimal_acc(X_test, y_test)
        optim_acc_array_train[PA_included][split_seed] = optimal_acc(X_train, y_train)

        # group C
        Group_C = y_pred_proba < 0.25
        group_c_rate_a = Group_C[group_test==1].mean() 
        group_c_rate_b = Group_C[group_test==2].mean()
        
        group_c_rate_array[PA_included][0][split_seed] = group_c_rate_a
        group_c_rate_array[PA_included][1][split_seed] = group_c_rate_b


    ### dump files
    performance_array = np.array([accuracy_array[PA_included], auroc_array[PA_included], 
                                  optim_acc_array[PA_included], optim_acc_array_train[PA_included]]).transpose()
    df_auroc_accuracy = pd.DataFrame(data=performance_array, 
                                     columns=['accuracy', 'auroc', 'optim_acc', 'optim_acc_train'])
    df_auroc_accuracy.to_csv(result_folder + f'/auroc_acc_{PA_included}.csv')

    df_group_c_rate = pd.DataFrame(data=group_c_rate_array[PA_included].transpose(), 
                                    columns=['male', 'female'])
    df_group_c_rate.to_csv(result_folder + f'/group_c_rates_{PA_included}.csv')


### dump inequality files
df_ineq = pd.DataFrame(data=pred_inequality_array.transpose(),
                        columns=['without PA', 'with PA'])
df_ineq.to_csv(result_folder + f'/pred_ineqs.csv')

df_di = pd.DataFrame(data=disp_imp_array.transpose(),
                        columns=['without PA', 'with PA'])
df_di.to_csv(result_folder + f'/disp_imps.csv')

# print single numbers
print(f'base rate diff {prot_feature}: {base_rate_diff}')
# gender : 0.035767008051107774
# citizen : 0.1660165321351032

print(f'proportion of group 1 in dataset: {group.mean() - 1}')
# proportion of group 1 (men) in dataset: 0.438145072940091
# proportion of group 1 (swiss) in dataset: 0.357204635876146