# Model_training.py
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script used train BRFC, LOGIT and LGBM on training set, and subsequently validation and testing it on validation and test sets.


import os
import warnings
import joblib  # For saving models
import optuna
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc, roc_curve, classification_report,
    log_loss, brier_score_loss, average_precision_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tqdm

# Suppress all warnings and LGBM verbose output
warnings.filterwarnings("ignore")

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Suppress warnings and LGBM output
warnings.filterwarnings("ignore")

# Define paths and create figures directory
figures_path = 'figures'
os.makedirs(figures_path, exist_ok=True)

# Load data
data_train_global = pd.read_csv(r'.../HTRAIN_Scaled.csv')
data_test_global = pd.read_csv(r'..../HVAL_Scaled.csv')
data_zmc_similar_global = pd.read_csv(r'.../HTEST_Scaled.csv')

# Separate features and outcome
feature_columns = list(data_train_global.columns)[:-3]
X_train = data_train_global[feature_columns].values
y_train = data_train_global['disability_progression'].values
X_val = data_test_global[feature_columns].values
y_val = data_test_global['disability_progression'].values
X_test = data_zmc_similar_global[feature_columns].values
y_test = data_zmc_similar_global['disability_progression'].values

# Define feature subsets
feature_subsets = {
    'Anatomical': [selected_volumetrics features from RFECV]
    'Radiomics': [selected_radiomics features from RFECV],
    'Anatomical + Radiomics': [selected_volumetrics and radiomics features from RFECV],
    'Clinical Superimposed': [selected_volumetrics and radiomics features from RFECV and clinical_features]
}

# Helper function to save PR and ROC plots
def save_pr_roc_curves(y_true, y_pred_proba, title, path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Calculate Youden's Index
    youden_index = tpr - fpr
    optimal_youden_index = np.argmax(youden_index)
    optimal_threshold_youden = thresholds[optimal_youden_index]

    # PR Curve
    plt.figure()
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f'{title} Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'{path}/{title}_PR.jpg')
    plt.close()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[optimal_youden_index], tpr[optimal_youden_index], color='red', label="Youden's Index")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f'{title} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{path}/{title}_ROC.jpg')
    plt.close()

    return pr_auc, roc_auc, optimal_threshold_youden

# Calculate precision, recall, sensitivity, and specificity
def calculate_metrics(y_true, y_pred_proba, threshold):
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return precision, recall, sensitivity, specificity


# Function for Optuna hyperparameter optimization
def objective(trial, model_type, X_train, y_train, X_val, y_val):
    weight_class_1 = trial.suggest_float("weight_class_1", 0.5, 3.0)
    class_weight = {0: 1.0, 1: weight_class_1}
    
    if model_type == 'Logistic Regression':
        C = trial.suggest_loguniform('C', 1e-4, 1e2)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', class_weight=class_weight,
            C=C, l1_ratio=l1_ratio, random_state=43, max_iter=1000
        )
    elif model_type == 'Balanced Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        model = BalancedRandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=43
        )
    elif model_type == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
        num_leaves = trial.suggest_int('num_leaves', 20, 40)
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            num_leaves=num_leaves, objective='binary', class_weight=class_weight,
            random_state=43, verbose=-1
        )
    
    model.fit(X_train, y_train)
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_val_pred_proba)


# Update `evaluate_model_with_cv` to save PR and ROC curves for train, validation, and test sets

def evaluate_model_with_cv(X_train, y_train, X_val, y_val, X_test, y_test, selected_features, model, feature_columns, cv_splits=10):
    # Convert selected feature names to indices
    selected_indices = [feature_columns.index(feature) for feature in selected_features]
    
    # Use these indices to slice the arrays
    X_train_sel, X_val_sel, X_test_sel = X_train[:, selected_indices], X_val[:, selected_indices], X_test[:, selected_indices]

    # Initialize lists to store metrics across folds
    train_pr_aucs, train_roc_aucs, val_pr_aucs, val_roc_aucs = [], [], [], []
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Stratified K-Fold cross-validation
    for train_index, val_index in skf.split(X_train_sel, y_train):
        X_train_cv, X_val_cv = X_train_sel[train_index], X_train_sel[val_index]
        y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

        # Fit the model on the cross-validation train split
        model.fit(X_train_cv, y_train_cv)

        # Training metrics for the current fold
        y_train_cv_pred = model.predict_proba(X_train_cv)[:, 1]
        train_precision, train_recall, _ = precision_recall_curve(y_train_cv, y_train_cv_pred)
        train_pr_auc_fold = np.trapz(train_recall, train_precision)
        train_roc_auc_fold = roc_auc_score(y_train_cv, y_train_cv_pred)
        train_pr_aucs.append(train_pr_auc_fold)
        train_roc_aucs.append(train_roc_auc_fold)
    
        # Validation metrics for the current fold
        y_val_cv_pred = model.predict_proba(X_val_cv)[:, 1]
        val_precision, val_recall, _ = precision_recall_curve(y_val_cv, y_val_cv_pred)
        val_pr_auc_fold = np.trapz(val_recall, val_precision)
        val_roc_auc_fold = roc_auc_score(y_val_cv, y_val_cv_pred)
        val_pr_aucs.append(val_pr_auc_fold)
        val_roc_aucs.append(val_roc_auc_fold)

    # Calculate mean and std for training and validation metrics across folds
    train_pr_auc_mean, train_pr_auc_std = np.mean(train_pr_aucs), np.std(train_pr_aucs)
    train_roc_auc_mean, train_roc_auc_std = np.mean(train_roc_aucs), np.std(train_roc_aucs)
    val_pr_auc_mean, val_pr_auc_std = np.mean(val_pr_aucs), np.std(val_pr_aucs)
    val_roc_auc_mean, val_roc_auc_std = np.mean(val_roc_aucs), np.std(val_roc_aucs)

    # Fit the model on the full training set and evaluate on the fixed test set
    model.fit(X_train_sel, y_train)

    # Save PR and ROC curves for training set
    y_train_pred = model.predict_proba(X_train_sel)[:, 1]
    train_pr_auc, train_roc_auc, train_optimal_youden = save_pr_roc_curves(
        y_train, y_train_pred, f"{model.__class__.__name__}_{subset_name}_Train", figures_path
    )

    # Save PR and ROC curves for validation set
    y_val_pred = model.predict_proba(X_val_sel)[:, 1]
    val_pr_auc, val_roc_auc, val_optimal_youden = save_pr_roc_curves(
        y_val, y_val_pred, f"{model.__class__.__name__}_{subset_name}_Validation", figures_path
    )

    # Save PR and ROC curves for test set
    y_test_pred = model.predict_proba(X_test_sel)[:, 1]
    test_pr_auc, test_roc_auc, optimal_youden_threshold = save_pr_roc_curves(
        y_test, y_test_pred, f"{model.__class__.__name__}_{subset_name}_Test", figures_path
    )

    # Additional metrics at Youden's threshold for test set
    precision_test, recall_test, sensitivity_test, specificity_test = calculate_metrics(y_test, y_test_pred, optimal_youden_threshold)
    precision_train, recall_train, sensitivity_train, specificity_train = calculate_metrics(y_train, y_train_pred, train_optimal_youden)
    precision_val, recall_val, sensitivity_val, specificity_val = calculate_metrics(y_val, y_val_pred, val_optimal_youden)

    # Return a dictionary of all relevant metrics
    return {
        "Train PR AUC Mean": train_pr_auc_mean,
        "Train PR AUC Std": train_pr_auc_std,
        "Train ROC AUC Mean": train_roc_auc_mean,
        "Train ROC AUC Std": train_roc_auc_std,
        "Train_Optimal Youden's Index Threshold_test": train_optimal_youden,
        "Precision Train at Youden_test": precision_train,
        "Recall Train at Youden_test": recall_train,
        "Sensitivity_Train": sensitivity_train,
        "Specificity_Train": specificity_train,
        "Validation PR AUC Mean": val_pr_auc_mean,
        "Validation PR AUC Std": val_pr_auc_std,
        "Validation ROC AUC Mean": val_roc_auc_mean,
        "Validation ROC AUC Std": val_roc_auc_std,
        "Val_Optimal Youden's Index Threshold_test": val_optimal_youden,
        "Precision VAL at Youden_test": precision_val,
        "Recall VAL at Youden_test": recall_val,
        "Sensitivity_val": sensitivity_val,
        "Specificity_val": specificity_val,
        "Test PR AUC": test_pr_auc,
        "Test ROC AUC": test_roc_auc,
        "Optimal Youden's Index Threshold_test": optimal_youden_threshold,
        "Precision at Youden_test": precision_test,
        "Recall at Youden_test": recall_test,
        "Sensitivity_test": sensitivity_test,
        "Specificity_test": specificity_test
    }

# Run Optuna tuning and evaluation

def objective(trial, model_type, X_train, y_train, X_val, y_val):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_auc_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Define and train model as before
        weight_class_1 = trial.suggest_float("weight_class_1", 0.5, 3.0)
        class_weight = {0: 1.0, 1: weight_class_1}
    
    if model_type == 'Logistic Regression':
        C = trial.suggest_loguniform('C', 1e-4, 1e2)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', class_weight=class_weight,
            C=C, l1_ratio=l1_ratio, random_state=43, max_iter=1000
        )
    elif model_type == 'Balanced Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        model = BalancedRandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=43
        )
    elif model_type == 'LightGBM':
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
        num_leaves = trial.suggest_int('num_leaves', 20, 40)
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            num_leaves=num_leaves, objective='binary', class_weight=class_weight,
            random_state=43, verbose=-1
        )

    # Perform cross-validation
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Train model
        model.fit(X_train_cv, y_train_cv)
        y_val_cv_pred = model.predict_proba(X_val_cv)[:, 1]
        
        pr_auc_scores.append(average_precision_score(y_val_cv, y_val_cv_pred))
    
    return np.mean(pr_auc_scores)



results = []
for model_name in ['Logistic Regression', 'Balanced Random Forest', 'LightGBM']:
    for subset_name, selected_features in feature_subsets.items():
        selected_indices = [feature_columns.index(feature) for feature in selected_features]
        X_train_sel, X_val_sel, X_test_sel = X_train[:, selected_indices], X_val[:, selected_indices], X_test[:, selected_indices]

        # Optuna hyperparameter optimization using cross-validation on the training set
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_name, X_train_sel, y_train), n_trials=50)
        best_params = study.best_params

        # Initialize the model with the best parameters
        if model_name == 'Logistic Regression':
            model = LogisticRegression(
                penalty='elasticnet', solver='saga', class_weight={0: 1, 1: best_params['weight_class_1']},
                C=best_params['C'], l1_ratio=best_params['l1_ratio'], random_state=43, max_iter=1000
            )
        elif model_name == 'Balanced Random Forest':
            model = BalancedRandomForestClassifier(
                n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'], min_samples_leaf=best_params['min_samples_leaf'],
                class_weight={0: 1, 1: best_params['weight_class_1']}, random_state=43
            )
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(
                n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'], num_leaves=best_params['num_leaves'],
                objective='binary', class_weight={0: 1, 1: best_params['weight_class_1']},
                random_state=43, verbose=-1
            )

        # Evaluate the final model on train, validation, and test sets
        eval_results = evaluate_model_with_cv(X_train, y_train, X_val, y_val, X_test, y_test, selected_features, model, feature_columns, cv_splits=5)

        # Save model and results
        joblib.dump(model, f"{model_name}_{subset_name}.pkl")
        results.append({"Model": model_name, "Subset": subset_name, "Parameters": best_params, **eval_results})

# Save results to CSV
summary_df = pd.DataFrame(results)
summary_df.to_csv('summary_results.csv', index=False)
print("Pipeline complete. Hyperparameter tuning used train CV only; validation set was used once.")

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows

# 2. Define the desired column order
desired_order = [
    'Model',
    'Subset',
    'Parameters',
    
    # Training Metrics
    'Train PR AUC Mean',
    'Train PR AUC Std',
    'Train ROC AUC Mean',
    'Train ROC AUC Std',
    'Validation PR AUC Mean',
    'Validation PR AUC Std',
    'Validation ROC AUC Mean',
    'Validation ROC AUC Std',
    'Test PR AUC',
    'Test ROC AUC',
    'Train_Optimal Youden\'s Index Threshold_test',
    'Precision Train at Youden_test',
    'Recall Train at Youden_test',
    'Sensitivity_Train',
    'Specificity_Train',
    
    # Validation Metrics

    'Val_Optimal Youden\'s Index Threshold_test',
    'Precision VAL at Youden_test',
    'Recall VAL at Youden_test',
    'Sensitivity_val',
    'Specificity_val',
    
    # Test Metrics
    'Optimal Youden\'s Index Threshold_test',
    'Precision at Youden_test',
    'Recall at Youden_test',
    'Sensitivity_test',
    'Specificity_test'
]

# Filter the desired_order to include only columns present in the DataFrame
existing_order = [col for col in desired_order if col in summary_df.columns]

summary_df = summary_df[existing_order]

summary_df.sort_values(by=['Validation PR AUC Mean'], ascending=False)

