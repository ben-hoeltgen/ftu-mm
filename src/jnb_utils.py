import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



def get_model(name):
    if name == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression())
    if name == 'GBM_opt':
        return GradientBoostingClassifier(n_estimators=500, max_depth=10, 
                                           max_leaf_nodes=50, min_samples_leaf=500)



def get_SCHL_thresholds(model, with_PA):
    
    """
    Compute decision thresholds for the SCHL (schooling/education level) feature 
    based on predicted probabilities from a model.

    Parameters
    ----------
    model : sklearn estimator
        A trained probabilistic classifier with a `predict_proba` method.
    with_PA : bool
        Whether to compute separate thresholds conditional on a protected attribute (PA).
        - If True: thresholds are computed separately for PA = 0 and PA = 1.
        - If False: a single threshold is computed (and returned twice for consistency).

    Returns
    -------
    thresholds : list of float
        List of two threshold values (one per PA group if `with_PA=True`, 
        otherwise the same threshold duplicated).
        Thresholds are rounded down to the nearest integer + 0.5.
    """
    SCHL_values = np.linspace(14, 24, 1000)
    if with_PA:
        thresholds = []
        for PA in [0, 1]:
            PA_values = np.ones(1000) + PA
            X = np.column_stack((SCHL_values, PA_values)).reshape(-1, 2)
            preds = model.predict_proba(X)[:, 1]
            thresh = 14 + (preds < 0.5).mean() * 10
            thresholds += [np.floor(thresh) + 0.5]
        return thresholds
    else:
        X = SCHL_values.reshape(-1, 1)
        preds = model.predict_proba(X)[:,1]
        thresh = 14 + (preds < 0.5).mean() * 10
        return [np.floor(thresh) + 0.5, np.floor(thresh) + 0.5]
    
    

def print_stats(y_test, yhat, yhat_probs, group_test):
    """
    Print performance & fairness statistics 
    return accuracy, auroc, and Disparate Impact
    """
    print('base rates:', y_test[group_test==1].mean(), y_test[group_test==2].mean())
    print('avg prediction:', yhat_probs[group_test==1].mean(), yhat_probs[group_test==2].mean())
    print('avg outcome:', yhat[group_test==1].mean(), yhat[group_test==2].mean())
    auc = roc_auc_score(y_test, yhat_probs)
    acc = accuracy_score(y_test, yhat)
    di = yhat[group_test==1].mean() - yhat[group_test==2].mean()
    print('auroc:', auc)
    print('accuracy:', acc)
    print('accuracy per group:', accuracy_score(y_test[group_test==1], yhat[group_test==1]),
      accuracy_score(y_test[group_test==2], yhat[group_test==2]))

    return acc, auc, di


### some functions to create plots 
### see acs_single_feature.ipynb


# helper function
def nan_or_mean(a):
    if len(a) == 0:
        return np.nan
    else:
        return a.mean()



def plot_calibration_curve(yhat_probs, y_test, group_test):
    # get averages per bin
    bin_middles = []
    bin_avg_preds_1 = []
    bin_avg_preds_2 = []
    bin_rates_1 = []
    bin_rates_2 = []
    for bin in range(10):
        bin_min = bin / 10; bin_max = (bin+1) / 10
        bin_middles = bin_middles + [bin_min + 0.05]
        bin_mask_1 = (yhat_probs >= bin_min) & (yhat_probs < bin_max) & (group_test==1)
        bin_mask_2 = (yhat_probs >= bin_min) & (yhat_probs < bin_max) & (group_test==2)

        bin_preds_1 = nan_or_mean(yhat_probs[bin_mask_1]) 
        bin_preds_2 = nan_or_mean(yhat_probs[bin_mask_2]) 
        bin_labels_1 = nan_or_mean(y_test[bin_mask_1]) 
        bin_labels_2 = nan_or_mean(y_test[bin_mask_2]) 
        
        bin_avg_preds_1 = bin_avg_preds_1 + [bin_preds_1]
        bin_avg_preds_2 = bin_avg_preds_2 + [bin_preds_2]
        bin_rates_1 = bin_rates_1 + [bin_labels_1]
        bin_rates_2 = bin_rates_2 + [bin_labels_2]

    # make plot
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), nrows=1, ncols=2)
    ax1.plot(bin_middles, bin_middles, '--', color='grey')
    ax1.plot(bin_avg_preds_1, bin_rates_1, label='group 1', marker='+', linestyle='', alpha=1, ms=13)
    ax1.plot(bin_avg_preds_2, bin_rates_2, label='group 2', marker='+', linestyle='', alpha=1, ms=13)
    ax1.set_xlabel('model prediction (binned)')
    ax1.xaxis.set_major_locator(MultipleLocator(.2))
    ax1.xaxis.set_minor_locator(MultipleLocator(.1))
    ax1.set_ylabel('average outcome')
    ax1.legend()

    ax2.hist(yhat_probs[group_test==1], alpha=0.5, bins=np.arange(10)/10)
    ax2.hist(yhat_probs[group_test==2], alpha=0.5, bins=np.arange(10)/10)
    ax2.xaxis.set_major_locator(MultipleLocator(.2))
    ax2.xaxis.set_minor_locator(MultipleLocator(.1))
    ax2.set_xlabel('model prediction (binned)')
    fig.tight_layout()
    return



def plot_x_calibration_curves(yhat_probs, group_test, X_test, y_test, thresholds, setting):
    # get bins
    SCHL_values = np.arange(1, 25) 
    bin_avg_preds_1 = []
    bin_avg_preds_2 = []
    bin_base_rates_1 = []
    bin_base_rates_2 = []

    for x_value in SCHL_values:
        bin_preds_1 = yhat_probs[(group_test==1) & (X_test[:,0] == x_value)]
        bin_preds_2 = yhat_probs[(group_test==2) & (X_test[:,0] == x_value)]
        #print('bin', bin, 'group 1:', len(bin_preds_1))
        #print('bin', bin, 'group 2:', len(bin_preds_2))
        bin_avg_preds_1 = bin_avg_preds_1 + [nan_or_mean(bin_preds_1)]
        bin_avg_preds_2 = bin_avg_preds_2 + [nan_or_mean(bin_preds_2)]
        # base rates per value
        bin_base_rates_1 = bin_base_rates_1 + [nan_or_mean(y_test[(group_test==1) & (X_test[:,0] == x_value)])]
        bin_base_rates_2 = bin_base_rates_2 + [nan_or_mean(y_test[(group_test==2) & (X_test[:,0] == x_value)])]

    # make plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(6, 3))

    # group 1
    ax1.plot(SCHL_values, [0.5] * len(SCHL_values), ':', color='black', label='50% threshold')
    ax1.plot(SCHL_values, bin_base_rates_1, '--', color='grey', label='base rate')
    ax1.plot(SCHL_values, bin_avg_preds_1, label='prediction')
    #ax1.set_xticks(SCHL_values)
    ax1.xaxis.set_major_locator(MultipleLocator(5))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.grid(axis='x', which='both', linewidth=0.3)
    ax1.vlines(thresholds[0], 0, 1, color='blue', linewidth=0.3)
    ax1.set_xlabel('education (SCHL)')
    ax1.legend()
    ax1.set_title(f'{setting} model, men')

    # group 2
    ax2.plot(SCHL_values, [0.5] * len(SCHL_values), ':', color='black', label='50% threshold')
    ax2.plot(SCHL_values, bin_base_rates_2, '--', color='grey', label='base rate')
    ax2.plot(SCHL_values, bin_avg_preds_2, color='orange', label='prediction')
    ax2.vlines(thresholds[1], 0, 1, color='red', linewidth=0.3)
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.grid(axis='x', which='both', linewidth=0.3)
    ax2.set_xlabel('education (SCHL)')
    ax2.legend()
    ax2.set_title(f'{setting} model, women')
    
    fig.tight_layout()
    plt.show()
    plt.close()

    # testset X distribution
    bin_corners = np.arange(1, 26) - 0.5
    fig, ax = plt.subplots(figsize=(4, 2.5))
    result1 = ax.hist(X_test[group_test==1][:,0], alpha=0.5, bins=bin_corners)
    result2 = ax.hist(X_test[group_test==2][:,0], alpha=0.5, bins=bin_corners)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    if thresholds[0] != thresholds[1]:
        ax.axvline(thresholds[0], color='blue', linewidth=0.5)
        ax.axvline(thresholds[1], color='red', linewidth=0.5)
    else:
        ax.axvline(thresholds[0], color='purple', linewidth=0.5)
        #plt.axvline(thresholds[0], color='blue', linewidth=0.5, linestyle='--')
    ax.set_xlabel('education (SCHL)')
    #print(X_test[group_test==1][:,0].mean(), X_test[group_test==2][:,1].mean())

    return
