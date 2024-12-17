# Feature_Selection.py
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script used balanced random forest as an estimator in RFECV to select feature subsets.



import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
print('sklearn version', sklearn.__version__)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
make_scorer, precision_score
)


# Ignore all warnings
warnings.filterwarnings("ignore")



X_train_global = pd.read_csv(r'.../HTRAIN_Scaled.csv')
X_val_global = pd.read_csv(r'.../HVAL_Scaled.csv')
data_zmc_global = pd.read_csv(r'..../HTEST_Scaled.csv')


# Since its shuffled, for the sake of pipeline, I am naming it zmc but it doesnt mean anything



def selectNonIntercorrelated(df_in, ftrs, corr_th, return_intercorrelated = False):
    
    # selection of the features, which are not 'highly intercorrelated' (correlation is defined by Spearman coefficient);
    # pairwise correlation between all the features is calculated, 
    # from each pair of features, considered as intercorrelated, 
    # feature with maximum sum of all the pairwise Spearman correlation coefficients is a 'candidate to be dropped'
    # for stability of the selected features, bootstrapping approach is used: 
    # in each bootstrap split, the random subsample, stratified in relation to outcome, 
    # is formed, based on original observations from input dataset;
    # in each bootstrap split, 'candidates to be dropped' are detected;
    # for each input feature, its frequency to appear as 'candidate to be dropped' is calculated,
    # features, appeared in 50 % of splits as 'candidate to be dropped', are excluded from feature set
    
    # input:
    # df_in - input dataframe, containing feature values (dataframe, columns = features, rows = observations),
    # ftrs - list of dataframe features, used in analysis (list of feature names - string variables),
    # corr_th - threshold for Spearman correlation coefficient, defining each pair of features as intercorrelated (float)
    
    # output:
    # non_intercorrelated_features - list of names of features, which did not appear as inter-correlated
    
    corr_matrix = df_in.corr(method='spearman').abs()
    mean_absolute_corr = corr_matrix.mean()
    intercorrelated_features_set = []
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    high_corrs = upper.where(upper > corr_th).dropna(how='all', axis=1).dropna(how='all', axis=0)

    for feature in high_corrs.columns:
        mean_absolute_main = mean_absolute_corr[feature]
        correlated_with_feature = high_corrs[feature].index[pd.notnull(high_corrs[feature])]
        for each_correlated_feature in correlated_with_feature:
            mean_absolute = mean_absolute_corr[each_correlated_feature]
            if mean_absolute_main > mean_absolute:
                if feature not in intercorrelated_features_set:
                    intercorrelated_features_set.append(feature)
            else:
                if each_correlated_feature not in intercorrelated_features_set:
                    intercorrelated_features_set.append(each_correlated_feature)

    non_intercorrelated_features_set = [e for e in ftrs if e not in intercorrelated_features_set] 
    #print ('Non intercorrelated features: ', non_intercorrelated_features_set)
    
    if return_intercorrelated:
        return non_intercorrelated_features_set, intercorrelated_features_set
    else:
        return non_intercorrelated_features_set


# ## Global Variables

features = list(X_train_global.columns)


features = features[:-3]



data_train_global = X_train_global[features].copy(deep=True)
data_test_global = X_val_global[features].copy(deep=True)
data_zmc_similar_global = data_zmc_global[features].copy(deep=True)
outcome_train_global = X_train_global['disability_progression'].copy(deep=True)
outcome_test_global = X_val_global['disability_progression'].copy(deep=True)
outcome_zmc_global = data_zmc_global['disability_progression'].copy(deep=True)


# ## 2. Radiomics Only (Lesion RF)


data_train = data_train_global
data_test = data_test_global
data_zmc_similar = data_zmc_similar_global
outcome_train = outcome_train_global
outcome_test = outcome_test_global
outcome_zmc = outcome_zmc_global


# Method 1: Chaining the drop commands without using in-place (preferred for readability and fewer errors)
#data_train = data_train.drop(columns=data_train.filter(like='CSF_').columns)\
#                                   .drop(columns=data_train.filter(like='THALAMUS_').columns)\
#                                   .drop(columns=data_train.filter(like='GM_').columns)


data_train = data_train.drop(columns=data_train.filter(like='_vol').columns)


data_train.drop(columns = ['Intra.Cranial', 'clinical_age', 'EDSS_T0', 'Gender_F', 'Gender_M', 'MRIpipeline_A', 
                           'MRIpipeline_B', 'MRIpipeline_CsT1', 'MRIpipeline_C', 'MRIpipeline_D', 'lstlpa.nLesions'], inplace = True)


features = list(data_train.columns)

features_non_intercorrelated, features_intercorrelated = selectNonIntercorrelated(data_train, features, 0.95, return_intercorrelated = True)
print ('Number of non-intercorrelated features: ', len(features_non_intercorrelated))

# print('Intercorrelated features: ', features_intercorrelated)

seed=11
run_rfe = True
if run_rfe:
    # Initialize SMOTE
    #smote = SMOTE(random_state=seed)

    # Apply SMOTE to the training data
    #data_smote, outcome_smote = smote.fit_resample(data_train[features_non_intercorrelated], outcome_train)
    #data_rfe = data_train[features_non_intercorrelated]
    #outcome_rfe = outcome_train
    data_rfe = data_train[features_non_intercorrelated]
    outcome_rfe = outcome_train
    
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=outcome_train)

    # Initialize the BRFC estimator
    #rfe_estimator = RandomForestClassifier(random_state=seed, class_weight={0: class_weights[0], 1: class_weights[1]})
    rfe_estimator = BalancedRandomForestClassifier(class_weight='balanced_subsample',
                                      random_state=seed+2,
                                      sampling_strategy = 'not majority',
                                      n_estimators = 25,
                                      max_depth = 10,
                                      min_samples_split = 5,
                                      min_samples_leaf=2
                                     )

    # Set up cross-validation
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.4, random_state=seed+3)
    scorer = make_scorer(precision_score)

    # Apply RFECV
    rfecv = RFECV(estimator=rfe_estimator, step=1, cv=cv, scoring=scorer, n_jobs=1)
    rfecv.fit(data_rfe, outcome_rfe)

    # Extracting cross-validation scores
    mean_scores = rfecv.cv_results_['mean_test_score']

    # Determine the optimal number of features
    optimal_num_features = rfecv.n_features_
    optimal_score = np.max(mean_scores)

    # Plotting the number of features vs. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel('Number of features selected')
    plt.ylabel('Mean cross-validation score')
    plt.scatter(range(1, len(mean_scores) + 1), mean_scores)
    plt.scatter(optimal_num_features, optimal_score, color='red', label=f'Optimal ({optimal_num_features} features)')
    plt.title('RFECV for Feature Selection')
    plt.legend()
    plt.show()

    print('Optimal number of features:', optimal_num_features)
    print('Optimal cross-validation score:', optimal_score)

    # Features selected by RFECV
    selected_features = rfecv.get_feature_names_out()

    # Feature importance for BRFC
    sel_feature_importance_dict = {selected_features[i]: rfecv.estimator_.feature_importances_[i] for i in range(len(selected_features))}

    # Sorting the selected features by their importance
    sorted_selected_features = sorted(sel_feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_selected_names = [feature for feature, score in sorted_selected_features]
    sorted_selected_scores = [score for feature, score in sorted_selected_features]

    # Plotting the scores of the top features
    n_top = 20
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_selected_names[:n_top], sorted_selected_scores[:n_top])
    plt.xlabel('Score')
    plt.ylabel(f'Top {n_top} Selected Features')
    plt.title(f'Top {n_top} Features Selected by RFECV')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.show()

    # Cumulative importance score
    threshold = 0.5
    cumulative_scores = np.cumsum(sorted_selected_scores)/np.sum(sorted_selected_scores)

    selected_features_set = [name for i, name in enumerate(sorted_selected_names) if cumulative_scores[i] <= threshold]
else:
    selected_features_set = features_non_intercorrelated


# ## 3. Volumetric Features

data_train = data_train_global
data_test = data_test_global
data_zmc_similar = data_zmc_similar_global
outcome_train = outcome_train_global
outcome_test = outcome_test_global
outcome_zmc = outcome_zmc_global


data_train = data_train.loc[:, ~data_train.columns.str.startswith('original_')]


data_train.drop(columns = ['Intra.Cranial'], inplace = True)


features = list(data_train.columns)


features_non_intercorrelated, features_intercorrelated = selectNonIntercorrelated(data_train, features, 0.95, return_intercorrelated = True)
print ('Number of non-intercorrelated features: ', len(features_non_intercorrelated))



run_rfe = True
if run_rfe:
    # Initialize SMOTE
    #smote = SMOTE(random_state=seed)

    # Apply SMOTE to the training data
    #data_smote, outcome_smote = smote.fit_resample(data_train[features_non_intercorrelated], outcome_train)
    #data_rfe = data_train[features_non_intercorrelated]
    #outcome_rfe = outcome_train
    data_rfe = data_train[features_non_intercorrelated]
    outcome_rfe = outcome_train
    
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=outcome_train)

    # Initialize the BRFC estimator
    #rfe_estimator = RandomForestClassifier(random_state=seed, class_weight={0: class_weights[0], 1: class_weights[1]})
    rfe_estimator = BalancedRandomForestClassifier(class_weight='balanced_subsample',
                                      random_state=seed+2,
                                      sampling_strategy = 'not majority',
                                      n_estimators = 25,
                                      max_depth = 10,
                                      min_samples_split = 5,
                                      min_samples_leaf=2
                                     )

    # Set up cross-validation
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.4, random_state=seed+3)
    scorer = make_scorer(precision_score)

    # Apply RFECV
    rfecv = RFECV(estimator=rfe_estimator, step=1, cv=cv, scoring=scorer, n_jobs=1)
    rfecv.fit(data_rfe, outcome_rfe)

    # Extracting cross-validation scores
    mean_scores = rfecv.cv_results_['mean_test_score']

    # Determine the optimal number of features
    optimal_num_features = rfecv.n_features_
    optimal_score = np.max(mean_scores)

    # Plotting the number of features vs. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel('Number of features selected')
    plt.ylabel('Mean cross-validation score')
    plt.scatter(range(1, len(mean_scores) + 1), mean_scores)
    plt.scatter(optimal_num_features, optimal_score, color='red', label=f'Optimal ({optimal_num_features} features)')
    plt.title('RFECV for Feature Selection')
    plt.legend()
    plt.show()

    print('Optimal number of features:', optimal_num_features)
    print('Optimal cross-validation score:', optimal_score)

    # Features selected by RFECV
    selected_features = rfecv.get_feature_names_out()

    # Feature importance for BRFC
    sel_feature_importance_dict = {selected_features[i]: rfecv.estimator_.feature_importances_[i] for i in range(len(selected_features))}

    # Sorting the selected features by their importance
    sorted_selected_features = sorted(sel_feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_selected_names = [feature for feature, score in sorted_selected_features]
    sorted_selected_scores = [score for feature, score in sorted_selected_features]

    # Plotting the scores of the top features
    n_top = 20
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_selected_names[:n_top], sorted_selected_scores[:n_top])
    plt.xlabel('Score')
    plt.ylabel(f'Top {n_top} Selected Features')
    plt.title(f'Top {n_top} Features Selected by RFECV')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.show()

    # Cumulative importance score
    threshold = 0.5
    cumulative_scores = np.cumsum(sorted_selected_scores)/np.sum(sorted_selected_scores)

    selected_features_set = [name for i, name in enumerate(sorted_selected_names) if cumulative_scores[i] <= threshold]
else:
    selected_features_set = features_non_intercorrelated


volumetric = selected_features_set


# ## RFE Radiomics and Anat

features_non_intercorrelated = [selected features from Radiomics and Volumetrics by RFECV]

data_train = data_train_global
data_test = data_test_global
data_zmc_similar = data_zmc_similar_global
outcome_train = outcome_train_global
outcome_test = outcome_test_global
outcome_zmc = outcome_zmc_global

run_rfe = True
if run_rfe:
    # Initialize SMOTE
    #smote = SMOTE(random_state=seed)

    # Apply SMOTE to the training data
    #data_smote, outcome_smote = smote.fit_resample(data_train[features_non_intercorrelated], outcome_train)
    #data_rfe = data_train[features_non_intercorrelated]
    #outcome_rfe = outcome_train
    data_rfe = data_train[features_non_intercorrelated]
    outcome_rfe = outcome_train
    
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=outcome_train)

    # Initialize the BRFC estimator
    #rfe_estimator = RandomForestClassifier(random_state=seed, class_weight={0: class_weights[0], 1: class_weights[1]})
    rfe_estimator = BalancedRandomForestClassifier(class_weight='balanced_subsample',
                                      random_state=seed+4,
                                      sampling_strategy = 'not majority',
                                      n_estimators = 25,
                                      max_depth = 10,
                                      min_samples_split = 5,
                                      min_samples_leaf=2
                                     )

    # Set up cross-validation
    cv = StratifiedShuffleSplit(n_splits=20, test_size=0.4, random_state=seed+5)
    scorer = make_scorer(precision_score)

    # Apply RFECV
    rfecv = RFECV(estimator=rfe_estimator, step=1, cv=cv, scoring=scorer, n_jobs=1)
    rfecv.fit(data_rfe, outcome_rfe)

    # Extracting cross-validation scores
    mean_scores = rfecv.cv_results_['mean_test_score']

    # Determine the optimal number of features
    optimal_num_features = rfecv.n_features_
    optimal_score = np.max(mean_scores)

    # Plotting the number of features vs. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel('Number of features selected')
    plt.ylabel('Mean cross-validation score')
    plt.scatter(range(1, len(mean_scores) + 1), mean_scores)
    plt.scatter(optimal_num_features, optimal_score, color='red', label=f'Optimal ({optimal_num_features} features)')
    plt.title('RFECV for Feature Selection')
    plt.legend()
    plt.show()

    print('Optimal number of features:', optimal_num_features)
    print('Optimal cross-validation score:', optimal_score)

    # Features selected by RFECV
    selected_features = rfecv.get_feature_names_out()

    # Feature importance for BRFC
    sel_feature_importance_dict = {selected_features[i]: rfecv.estimator_.feature_importances_[i] for i in range(len(selected_features))}

    # Sorting the selected features by their importance
    sorted_selected_features = sorted(sel_feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    sorted_selected_names = [feature for feature, score in sorted_selected_features]
    sorted_selected_scores = [score for feature, score in sorted_selected_features]

    # Plotting the scores of the top features
    n_top = 20
    plt.figure(figsize=(12, 8))
    plt.barh(sorted_selected_names[:n_top], sorted_selected_scores[:n_top])
    plt.xlabel('Score')
    plt.ylabel(f'Top {n_top} Selected Features')
    plt.title(f'Top {n_top} Features Selected by RFECV')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score at the top
    plt.show()

    # Cumulative importance score
    threshold = 0.5
    cumulative_scores = np.cumsum(sorted_selected_scores)/np.sum(sorted_selected_scores)

    selected_features_set = [name for i, name in enumerate(sorted_selected_names) if cumulative_scores[i] <= threshold]
else:
    selected_features_set = features_non_intercorrelated


