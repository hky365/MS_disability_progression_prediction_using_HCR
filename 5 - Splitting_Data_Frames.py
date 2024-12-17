# Radiomics Features Extraction
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script Splits the data frame into training Validation and test set.

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


# Ignore all warnings
warnings.filterwarnings("ignore")


pelt_H = pd.read_csv(r'../path_to_lesion_and_other_ROI_radiomics_features_and_volume_features(if any).../Lesions_NAWM_combined.csv')
zmc_H = pd.read_csv(r'../path_to_lesion_and_other_ROI_radiomics_features_and_volume_features(if any).../Lesions_NAWM_combined.csv')

def encode_features(df, col, ONEHOT=True):
    if ONEHOT:
        dummies = pd.get_dummies(df[col], prefix=col)
        df.drop(col, axis='columns', inplace=True)
        df = pd.concat([df, dummies], axis=1)    
    return df


#cols_needed = list(pelt_H.columns)


pelt_H.rename(columns = {'PATID':'clinic_id', 'EDSS_t0':'EDSS_T0'}, inplace=True)

zmc_H['ID'] = zmc_H['clinic_id']

pelt_H.drop([ 't0', 'Fifth.Ventricle_vol'], axis=1, inplace=True)


# ## Now Splitting them
desired_start_columns = [
    'Index_NH', 'clinic_id','ID', 'Session', 'MRIpipeline', 
    'MRIdate', 'clinical_age', 'Gender', 'EDSS_T0', 'disability_progression'
]

reordered_columns = desired_start_columns + [col for col in pelt_H.columns if col not in desired_start_columns]

pelt_H.rename(columns={'disability_progression':'disability_progression_a'}, inplace=True)
pelt_H['disability_progression'] = pelt_H['disability_progression_a']

pelt_H.drop(['disability_progression_a'], axis=1, inplace=True)

# Applying the new column order to the dataframe
pelt_H = pelt_H[reordered_columns]

zmc_H = zmc_H[pelt_H.columns]

for i in pelt_H.columns:
    print(i)


# Assume 'pelt_H' is the dataset you're working with, and 'clinic_id' is the patient identifier
# Step 1: Get the unique patients
patients = pelt_H['clinic_id'].unique()

# Step 2: Group by 'clinic_id' to ensure all sessions for a patient are kept together
patient_groups = pelt_H.groupby('clinic_id')

# Step 3: Create a DataFrame with one row per patient for splitting purposes
patient_df = pd.DataFrame({
    'clinic_id': patients,
    'disability_progression': [patient_groups.get_group(pid)['disability_progression'].iloc[0] for pid in patients]
})

# Step 4: Split into Train and Validation sets, ensuring stratification by 'disability_progression'
train_patients, val_patients = train_test_split(patient_df, test_size=0.3, stratify=patient_df['disability_progression'], random_state=42)

# Step 5: Create the final Train and Validation DataFrames
train_df = pelt_H[pelt_H['clinic_id'].isin(train_patients['clinic_id'])]
val_df = pelt_H[pelt_H['clinic_id'].isin(val_patients['clinic_id'])]

# Step 6: Check the splits
print(f"Train set patients: {train_df['clinic_id'].nunique()}")
print(f"Validation set patients: {val_df['clinic_id'].nunique()}")

# Optional: Check the distribution of disability_progression in each split
print("Train set disability_progression distribution:")
print(train_df['disability_progression'].value_counts(normalize=True))

print("Validation set disability_progression distribution:")
print(val_df['disability_progression'].value_counts(normalize=True))


data_train = train_df.copy(deep=True)
data_test = val_df.copy(deep=True)

data_train.to_csv(r'.../HTRAIN.csv', index=False)
data_test.to_csv(r'.../HVAL.csv', index=False)
zmc_H.to_csv(r'...../HZMC.csv', index=False)

X_train_global_H = data_train.copy(deep=True)
X_val_global_H = data_test.copy(deep=True)
data_train = X_train_global_H.copy(deep=True)
data_test = X_val_global_H.copy(deep=True)

# target labels
outcome_train = data_train['disability_progression']
outcome_test = data_test['disability_progression']

data_zmc_similar = zmc_H[data_train.columns]

# Step 1: Store the outcome and other columns that need to be added back after scaling
outcome_cols_train = data_train[['disability_progression', 'clinic_id', 'Session']].reset_index(drop=True)
outcome_cols_test = data_test[['disability_progression', 'clinic_id', 'Session']].reset_index(drop=True)
outcome_cols_zmc = data_zmc_similar[['disability_progression', 'clinic_id', 'Session']].reset_index(drop=True)

# Encode MRI pipeline as dummy variables
data_train = encode_features(data_train, 'Gender')
data_test = encode_features(data_test, 'Gender')

zmc_H = encode_features(zmc_H, 'Gender')

# Step 2: Remove these columns from the dataset to avoid scaling them
features = list(data_train.columns[:])
features.remove('disability_progression')
#features.remove('Fifth.Ventricle_vol')
#features.remove('Age_in_months')
#features.remove('t0')
features.remove('ID')
features.remove('Index_NH')
features.remove('MRIdate')
#features.remove('MRIdate_final')
#features.remove('OAZIS_PATID')
features.remove('Session')
features.remove('segmentation_lesions_binarised')
features.remove('segmentation_lesions')
features.remove('segmentation_all')
features.remove('nawm_mask')
features.remove('raw_flair')
features.remove('normalised_image')
features.remove('clinic_id')

# Select features of interest
data_train = data_train[features]
data_test = data_test[features]

# Encode MRI pipeline as dummy variables
data_train = encode_features(data_train, 'MRIpipeline')
data_test = encode_features(data_test, 'MRIpipeline')

# Categorical and numerical features
categorical_features = ['MRIpipeline_A', 'MRIpipeline_B', 'MRIpipeline_C', 'MRIpipeline_CsT1', 'MRIpipeline_D', 'Gender_M', 'Gender_F']

# Numerical features by excluding categorical ones
numerical_features = [col for col in data_train.columns if col not in categorical_features]

# Apply the same to the zmc_similar dataset
data_zmc_similar = zmc_H[features]
data_zmc_similar = encode_features(data_zmc_similar, 'MRIpipeline')
data_zmc_similar['MRIpipeline_B'] = False
data_zmc_similar['MRIpipeline_C'] = False
data_zmc_similar['MRIpipeline_CsT1'] = False
data_train['MRIpipeline_D'] = False
data_test['MRIpipeline_D'] = False

# Step 3: Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
data_train_scaled = scaler.fit_transform(data_train[numerical_features])

# Transform the validation data using the fitted scaler
data_test_scaled = scaler.transform(data_test[numerical_features])

# Re-build dataframes
data_train_scaled = pd.DataFrame(data_train_scaled, columns=data_train[numerical_features].columns).reset_index(drop=True)
data_train = pd.concat([data_train_scaled, data_train[categorical_features].reset_index(drop=True)], axis=1)

data_test_scaled = pd.DataFrame(data_test_scaled, columns=data_test[numerical_features].columns).reset_index(drop=True)
data_test = pd.concat([data_test_scaled, data_test[categorical_features].reset_index(drop=True)], axis=1)

# Step 4: Attach the outcome, clinic_id, and Session columns back
data_train = pd.concat([data_train, outcome_cols_train], axis=1)
data_test = pd.concat([data_test, outcome_cols_test], axis=1)

# Checking the shape after transformation
print(f"Train set shape: {data_train.shape}")
print(f"Validation set shape: {data_test.shape}")

# Step 5: Scale the 'data_zmc_similar' and rebuild the dataframe
data_zmc_similar_scaled = scaler.transform(data_zmc_similar[numerical_features])
data_zmc_similar_scaled = pd.DataFrame(data_zmc_similar_scaled, columns=data_zmc_similar[numerical_features].columns).reset_index(drop=True)

# Concatenate the scaled numerical features with the original categorical features
data_zmc_similar = pd.concat([data_zmc_similar_scaled, data_zmc_similar[categorical_features].reset_index(drop=True)], axis=1)

# Attach the outcome, clinic_id, and Session columns back for 'data_zmc_similar'
data_zmc_similar = pd.concat([data_zmc_similar, outcome_cols_zmc], axis=1)

# Checking the shape after transformation
print(f"data_zmc_similar shape: {data_zmc_similar.shape}")

data_train.to_csv(r'.../HTRAIN_Scaled.csv', index=False)
data_test.to_csv(r'..../HVAL_Scaled.csv', index=False)
data_zmc_similar.to_csv(r'.../HZMC_Scaled.csv', index=False)


