# Delta_dataframe.py
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script used scaled train, test and validation sets to make delta data frames.


import pandas as pd
import numpy as np

X_train_global = pd.read_csv(r'..../HTRAIN_Scaled.csv')
X_val_global = pd.read_csv(r'..../HVAL_Scaled.csv')
data_zmc_global = pd.read_csv(r'.../HTEST_Scaled.csv')


# Final disability progression is from T1
def make_triplets(df):
    df = df.copy()
    new_df_rows = []

    # Convert the Session column to datetime format (assuming it's in YYYYMMDD format)
    df['Session'] = df['Session'].astype('str').str.zfill(8)
    df['Session'] = pd.to_datetime(df['Session'], format='%Y%m%d', errors='coerce')

    # Sort data by Session date
    df = df.sort_values(by='Session')

    # Iterate over the rows in the dataframe
    for patid, patient_group in df.groupby('clinic_id'):
        print("-----------------------------------------------")
        print(f"Patient ID: {patid}, Total Sessions: {len(patient_group)}")
        triplet_count = 0

        # Sort the patient's sessions in chronological order
        patient_group = patient_group.sort_values(by='Session')

        # Iterate over each session (T0) for the patient
        for idx, (_, row) in enumerate(patient_group.iterrows()):

            # Define T0
            t0_session = row['Session']
            
            # Define the 6-month anchor
            anchor_date = t0_session + pd.Timedelta(180, 'd')  # 6 months from T0
            
            # Define the window of ±3 months around the anchor
            window_start = anchor_date - pd.Timedelta(90, 'd')  # 3 months before anchor
            window_end = anchor_date + pd.Timedelta(90, 'd')  # 3 months after anchor

            # Find future sessions that could be T1 (after T0)
            remaining_rows = patient_group.iloc[idx+1:]

            # Loop through the remaining sessions to find T1 within the window
            for _, next_row in remaining_rows.iterrows():
                next_session = next_row['Session']

                # Check if the next session (T1) falls within the ±3 month window around the anchor
                if window_start <= next_session <= window_end:
                    print(f"T0: {t0_session}, Anchor: {anchor_date}, T1: {next_session}, Within ±3 month window")
                    
                    # Use the disability progression from T1 for consistency in delta radiomics
                    combined_row = {**row.add_suffix('_T0').to_dict(), **next_row.add_suffix('_T1').to_dict()}

                    # Ensure the disability_progression is taken from T1
                    combined_row.update({
                        'clinic_id': row['clinic_id'],
                        'disability_progression': next_row['disability_progression'],  # Use T1 progression
                        'clinical_Gender_F': row['Gender_F'],
                        'clinical_Gender_M': row['Gender_M'],
                        'EDSS_T0': row['EDSS_T0'],
                        'Session_T0': row['Session'],
                        'Session_T1': next_row['Session']
                    })
                    
                    # Append the combined row to the list
                    new_df_rows.append(combined_row)
                    triplet_count += 1

        print(f"Patient ID {patid}: {triplet_count} triplets created\n")

    return pd.DataFrame.from_records(new_df_rows)

triplets_X_train = make_triplets(X_train_global)
triplets_data_zmc = make_triplets(data_zmc_global)
triplets_X_val = make_triplets(X_val_global)

def create_relative_delta_radiomics_df(triplets_df):
    '''
    Creates a dataframe containing the relative delta values ((T1 - T0) / T0) for relevant features,
    while excluding specific columns and ensuring consistency in clinic_id and gender.

    Parameters:
    triplets_df (pd.DataFrame): The dataframe with T0 and T1 values.

    Returns:
    pd.DataFrame: A dataframe with the relative deltas for each relevant feature.
    '''
    # Copy the dataframe to avoid modifying the original
    triplets_df = triplets_df.copy()

    # List of columns to exclude from delta calculations
    exclude_columns = [
        'EDSS_T0_T0', 'MRIpipeline_A_T0', 'MRIpipeline_B_T0',
        'MRIpipeline_C_T0', 'MRIpipeline_CsT1_T0', 'MRIpipeline_D_T0',
        'Gender_M_T0', 'Gender_F_T0', 'disability_progression_T0',
        'clinic_id_T0', 'Session_T0'
    ]
    
    # Create a new dataframe for relative delta features
    relative_delta_features = {}
    
    # Iterate over all columns to calculate relative deltas for T0 and T1 pairs
    for col in triplets_df.columns:
        if '_T0' in col and col not in exclude_columns:
            base_feature = col.replace('_T0', '')
            t1_col = base_feature + '_T1'
            
            # Check if the corresponding T1 column exists and is numeric
            if t1_col in triplets_df.columns:
                if pd.api.types.is_numeric_dtype(triplets_df[col]) and pd.api.types.is_numeric_dtype(triplets_df[t1_col]):
                    # Calculate relative delta ((T1 - T0) / T0)
                    delta_col_name = 'delta_' + base_feature
                    t0_values = triplets_df[col]
                    t1_values = triplets_df[t1_col]
                    
                    # Avoid division by zero by replacing zero T0 values with NaN
                    with np.errstate(divide='ignore', invalid='ignore'):
                        relative_delta = (t1_values - t0_values) / t0_values
                        relative_delta = relative_delta.replace([np.inf, -np.inf], np.nan)
                    
                    relative_delta_features[delta_col_name] = relative_delta
                else:
                    print(f"Skipping non-numeric column: {col} and {t1_col}")
    
    # Convert the dictionary of relative delta features to a DataFrame
    relative_delta_df = pd.DataFrame(relative_delta_features)

    # Keep necessary identifying columns such as clinic_id, disability_progression, and sessions
    relative_delta_df['clinic_id'] = triplets_df['clinic_id_T0']  # Use clinic_id_T0 for identification
    relative_delta_df['disability_progression'] = triplets_df['disability_progression_T0']
    relative_delta_df['Session_T0'] = triplets_df['Session_T0']
    relative_delta_df['Session_T1'] = triplets_df['Session_T1']
    relative_delta_df['Gender_F'] = triplets_df['Gender_F_T0']

    # Ensure consistency between clinic_id_T0 and clinic_id_T1
    if not (triplets_df['clinic_id_T0'] == triplets_df['clinic_id_T1']).all():
        print("Warning: Mismatch between clinic_id_T0 and clinic_id_T1")

    # Ensure consistency between Gender_M and Gender_F
    if not (triplets_df['Gender_M_T0'] == triplets_df['Gender_M_T1']).all():
        print("Warning: Mismatch in Gender_M between T0 and T1")
    if not (triplets_df['Gender_F_T0'] == triplets_df['Gender_F_T1']).all():
        print("Warning: Mismatch in Gender_F between T0 and T1")

    return relative_delta_df

delta_train = create_relative_delta_radiomics_df(triplets_X_train)

delta_val = create_relative_delta_radiomics_df(triplets_X_val)

delta_zmc = create_relative_delta_radiomics_df(triplets_data_zmc)


delta_zmc.to_csv(r'..../Scaled_relative_TRIPLETS.csv', index=False)
delta_train.to_csv(r'.../Scaled_relative_TRIPLETS.csv', index=False)
delta_val.to_csv(r'.../Scaled_relative_TRIPLETS.csv', index=False)

# For Delta - 6 - Feature_Selection.py, 7 - Model_training.py should be repeated with input as delta_zmc, delta_train and delta_val
