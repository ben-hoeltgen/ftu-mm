import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from folktables import ACSDataSource, ACSIncome, ACSEmployment, BasicProblem


### function to create dataset from ACS data

def create_acs_dataset(task, state, income_threshold, include_protected, protected_att='SEX', 
                   only_features=None):
    """
    Create a dataset from ACS data using the folktables package.

    Parameters
    ----------
    task : str
        Prediction task to set up. Options: 'income' or 'employment'
    state : str
        State abbreviation (e.g., 'NY'). If 'ALL', data from all states is used.
    income_threshold : int
        Income cutoff used when task == 'income'. Default is 50k.
    include_protected : bool
        Whether to include the protected attribute in the feature set.
    protected_att : str, optional (default='SEX')
        If 'RAC1P', the dataset is filtered to only include datapoint with attribute Black or White 
        (largest groups in ALL and NY, but not e.g. in CA).
    only_features : list of str, optional
        If provided, restricts features to the provided list `only_features` (plus potentially protected_att)

    Returns
    -------
    features : np.ndarray
        Feature matrix (n_samples x n_features).
    label : np.ndarray
        Target variable (n_samples,).
    group : np.ndarray
        Binary attribute indicating group memberhip based on `protected_att` (n_samples,).
    """
    states = None if state == 'ALL' else [state] # give either list of states (one) or None for all

    # load data
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person', root_dir="data/folktables")
    acs_data = data_source.get_data(states=states, download=False)
    
    # binarise race attribute: only look at black and white
    if protected_att == 'RAC1P':
        acs_data = acs_data[(acs_data['RAC1P'] == 1.0) | (acs_data['RAC1P'] == 2.0)]

    # set up prediction problem: prelims
    if task == 'income':
        BaseProblem = ACSIncome 
        target_transform = lambda x: x > income_threshold
    elif task == 'employment':
        BaseProblem = ACSEmployment
        target_transform = lambda x: x == 1
    else:
        ValueError('task must be income or employment')

    # adapt list of used features
    feature_list = BaseProblem._features.copy()
    
    if protected_att not in feature_list:
        print(ACSIncome._features)
        print(BaseProblem._features)
        print(protected_att, feature_list)
        ValueError('protected attribute must be in features!' )

    if only_features is not None:
        feature_list = list(set(feature_list) & (set(only_features))) + [protected_att]

    if not include_protected:
        feature_list.remove(protected_att)

    # set up prediction problem: final
    CustomProblem = BasicProblem(
        features=feature_list,
        target=BaseProblem._target,
        target_transform=target_transform,    
        group=protected_att,
        preprocess=BaseProblem._preprocess,
        postprocess=BaseProblem._postprocess,
    )

    ### get numpy arrays
    features, label, group = CustomProblem.df_to_numpy(acs_data)

    print('dataset size:', features.shape)

    return features, label, group




### function to create dataset from Swiss Labor Market data
def create_almp_dataset(protected_att='female', include_protected=True):
    """
    Create a dataset from the Swiss Labor Market ALMP (Active Labor Market Programs) data.
    https://www.swissubase.ch/en/catalogue/studies/13867/latest/datasets/1203/1953/overview
    Data loaded locally from `data/swiss_labor_market/ALMP_Data.csv`.

    Parameters
    ----------
    protected_att : str, optional (default='female')
        Protected attribute to define the sensitive group. Options: 'female', 'swiss'
    include_protected : bool, optional (default=True)
        Whether to include the protected attribute as a feature in the dataset.

    Returns
    -------
    features : np.ndarray
        Feature matrix (n_samples x n_features). 
    label : np.ndarray
        Binary target variable indicating employment status:
        1 if employed for at least 6 of the first 24 months after program start, else 0  (n_samples,).
    group : np.ndarray
        Binary attribute indicating group memberhip based on `protected_att` (n_samples,).
    """
    data = pd.read_csv('data/swiss_labor_market/ALMP_Data.csv')

    job_trainings = ['computer3', 'computer6', 'emp_program3', 'emp_program6', 'job_search3', 'job_search6',
                'vocational3', 'vocational6', 'personality3', 'personality6']
    feature_list = ['qual', 'female', 'married', 'age', 'swiss', 'city',  'emp_share_last_2yrs', 
                'ue_spells_last_2yrs', 'emp_spells_5yrs', 'gdp_pc', 'unemp_rate', 
                'prev_job_manager','prev_job_sec_mis', 'prev_job_sec1', 'prev_job_sec2', 
                'prev_job_sec3', 'prev_job_self', 'prev_job_skilled', 'prev_job_unskilled']
    
    feature_list += job_trainings

    # restrict to German speaking cantons (as in Knaus double ML paper), improves acc from 0.63 to 0.65
    data = data.drop(data[data['canton_german'] == 0].index) 

    # pre-process qualification feature
    data['qual'] = data['qual'].map({'unskilled': 0.0, 'semiskilled': 1.0, 'no degree': 2.0, 'With degree': 3.0})
    
    # unaware vs aware
    if not include_protected:
        feature_list.remove(protected_att)

    # create feature array
    df_features = data[feature_list]
    features = np.nan_to_num(df_features.to_numpy(), -1)

    print(feature_list)

    # create group array
    if protected_att == 'female':
        group = data['female'] + 1 # 1 for male, 2 for female
    elif protected_att == 'swiss':
        group = 2 - data['swiss'] # 1 for Swiss, 2 for non-Swiss
    else:
        raise ValueError('Invalid protected attribute')
    
    # create label array
    months_employed = data['employed24']
    for month in range(1, 24):
        months_employed = months_employed + data[f'employed{month}']
    label = (months_employed >= 6).astype(int)

    return features, label, group




### function to fit model to selected features
def fit_model(X_train, y_train, model_class = 'LR'):
    """
    Fit a predictive model to the training data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (n_samples x n_features).
    y_train : np.ndarray
        Binary target labels corresponding to X_train.
    model_class : str, optional (default='LR')
        Choice of model:
            - 'LR': Logistic Regression with standardization.
            - 'GBM': Gradient Boosting (default GBM version).
            - 'GBM_opt': Optimized Gradient Boosting (high accuracy, based on tuning in to Cruz et al.).

    Returns
    -------
    model : sklearn estimator
        Fitted model pipeline/classifier ready for prediction.
    """
    if model_class == 'LR':
        model = make_pipeline(StandardScaler(), LogisticRegression())
    elif model_class == 'GBM': # standard lightweight version
        model = GradientBoostingClassifier(n_estimators=10, max_depth=5)
    elif model_class == 'GBM_opt': # optimal accuracy, similar to best model in Cruz paper on Income
        model = GradientBoostingClassifier(n_estimators=500, max_depth=10, 
                                           max_leaf_nodes=50, min_samples_leaf=500)
    else:
        raise ValueError('model_class must be LR or GBM')
    model.fit(X_train, y_train)
    return model



### define function to calculate the accuracy of the bayesian optimal classifier for a given dataset

def optimal_acc(X, y):
    """
    Compute the maximum achievable classification accuracy on a dataset,
    based on the empiical distribution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples x n_features).
    y : np.ndarray
        Binary labels (0/1).

    Returns
    -------
    acc : float
        The optimal achievable accuracy score (between 0 and 1).
    """
    X_df = pd.DataFrame(X)
    # Convert rows to X (hashable format for grouping)
    X_df['X_tuple'] = X_df.apply(tuple, axis=1)
    # Combine with y_test into a DataFrame
    df = pd.DataFrame({'X_tuple': X_df['X_tuple'].values, 'y': y})
    # Compute label ratio (mean y per unique X)
    label_ratios = df.groupby('X_tuple')['y'].mean().reset_index()
    # Rename for clarity
    label_ratios.columns = ['X_tuple', 'label_ratio']
    # Apply threshold (≥ 0.5 → 1, otherwise → 0)
    label_ratios['predicted_label'] = (label_ratios['label_ratio'] >= 0.5).astype(int)
    # Map predictions back to X_test
    X_df['predicted_label'] = X_df['X_tuple'].map(label_ratios.set_index('X_tuple')['predicted_label'])

    return accuracy_score(y, X_df['predicted_label'])

### calculate accuracy on test set of the classifier thats best on train+test set together 
# (closer to TD intuition but MM bounds do not apply)

def optimal_acc_test(X_train, y_train, X_test, y_test):
    """
    Compute test accuracy of the optimal classifier trained on combined
    train+test data.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    acc : float
        Accuracy of the optimal classifier evaluated only on the test set.
    """
    X = np.concatenate((X_train, X_test))#
    y = np.concatenate((y_train, y_test))

    X_df = pd.DataFrame(X)
    # Convert rows to X (hashable format for grouping)
    X_df['X_tuple'] = X_df.apply(tuple, axis=1)
    # Combine with y_test into a DataFrame
    df = pd.DataFrame({'X_tuple': X_df['X_tuple'].values, 'y': y})
    # Compute label ratio (mean y per unique X)
    label_ratios = df.groupby('X_tuple')['y'].mean().reset_index()
    # Rename for clarity
    label_ratios.columns = ['X_tuple', 'label_ratio']
    # Apply threshold (≥ 0.5 → 1, otherwise → 0)
    label_ratios['predicted_label'] = (label_ratios['label_ratio'] >= 0.5).astype(int)

    # Map predictions back to X_test
    X_test_df = pd.DataFrame(X_test)
    X_test_df['X_tuple'] = X_test_df.apply(tuple, axis=1)
    X_test_df['predicted_label'] = X_test_df['X_tuple'].map(label_ratios.set_index('X_tuple')['predicted_label'])

    return accuracy_score(y_test, X_test_df['predicted_label'])
