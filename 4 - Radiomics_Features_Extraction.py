# Radiomics Features Extraction
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script extracts radiomics features from ROI.


import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
#from statsmodels.stats.multitest import multipletests
import os
import glob
import nibabel as nib
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations


#base_dir = r'.../Path_to_ROI_Masks/..'

# ## Lesion Radiomics Extraction



# Instantiate the extractor with the parameter file
params = r'path_to_parameter_file\exampleMR_1mm.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# Create an empty DataFrame to store the features
df = pd.DataFrame()

# Create an empty list to store problematic IDs and sessions
problematic_ids_sessions = []

# Initialize tqdm with the total number of rows in the CSV
pbar = tqdm(total=len(missing_df), desc='Processing features')

# Loop over each row in the CSV to construct paths
for idx, row in missing_df.iterrows():
    subject = row['Subject.folder']
    session = row['Session.folder']
    
    # Construct the path to the normalized image and the wml_mask
    normalized_image_path = os.path.join(base_dir, subject, session, 'anat', 'normalized_image_final.nii.gz')
    wml_mask_path = os.path.join(base_dir, subject, session, 'anat', 'wml_mask.nii.gz')

    # Check if both the normalized image and wml_mask exist
    if not os.path.exists(normalized_image_path) or not os.path.exists(wml_mask_path):
        print(f'Files missing for {subject}/{session}. Skipping this directory...')
        problematic_ids_sessions.append(f'{subject}/{session}')
        pbar.update(1)
        continue

    # Load the lesion mask (wml_mask.nii.gz)
    mask_nifti = nib.load(wml_mask_path)
    mask_data = mask_nifti.get_fdata()

    print(f'Unique values in the mask for {subject}/{session}:', np.unique(mask_data))

    # Binarize the mask if necessary
    if len(np.unique(mask_data)) > 2 or not np.all(np.isin(np.unique(mask_data), [0, 1])):
        print(f'Binarizing the mask for {subject}/{session}...')
        
        # Replace NaN values with 0
        mask_data = np.nan_to_num(mask_data)
        
        # Binarize the mask
        threshold = 0.0
        mask_data_binarized = np.where(mask_data > threshold, 1, 0)

        # Check for problematic mask
        if np.sum(mask_data_binarized) <= 1:
            print(f'Problematic mask encountered for {subject}/{session}. Skipping this directory...')
            problematic_ids_sessions.append(f'{subject}/{session}')
            pbar.update(1)
            continue

        # Preserve the original mask's physical space attributes
        mask_nifti_binarized = nib.Nifti1Image(mask_data_binarized.astype(np.int16), mask_nifti.affine, mask_nifti.header)

        # Save the binarized mask
        binarized_mask_nifti_path = wml_mask_path.replace('.nii.gz', '_binarized_corrected.nii.gz')
        nib.save(mask_nifti_binarized, binarized_mask_nifti_path)

        # Use the binarized mask for feature extraction
        wml_mask_path = binarized_mask_nifti_path

    # Extract features from the normalized image and lesion mask
    try:
        result = extractor.execute(normalized_image_path, wml_mask_path)
    except ValueError as e:
        print(f'Problematic mask encountered for {subject}/{session} with error: {str(e)}')
        problematic_ids_sessions.append({
            "Subject": subject,
            "Session": session,
            "Error": str(e)
        })
        pbar.update(1)
        continue

    # Convert the result to a DataFrame
    result_df = pd.DataFrame.from_dict(result, orient='index').T

    # Store the ID and session in the DataFrame
    result_df['ID'] = subject.replace('sub-', '')  # Extracts ID and removes 'sub-' prefix
    result_df['Session'] = session.replace('ses-', '')  # Extracts Session and removes 'ses-' prefix

    # Append the DataFrame to the main DataFrame
    df = df.append(result_df, ignore_index=True)

    # Update the progress bar
    pbar.update(1)

pbar.close()  # Close the progress bar

# Save the DataFrame with extracted features to a CSV file
csv_file = os.path.join(base_dir, 'NOLST_wml_radiomics_features_corrected.csv')
df.to_csv(csv_file, index=False)

# Print the DataFrame
print(df)

# Save problematic IDs and sessions to a DataFrame and then to a CSV file
problematic_df = pd.DataFrame(problematic_ids_sessions, columns=['Problematic Session'])
problematic_csv_file = os.path.join(base_dir, 'NOLST_wml_problematic_ids_sessions_corrected.csv')
problematic_df.to_csv(problematic_csv_file, index=False)

print('Problematic IDs and Sessions saved to CSV.')

## Repeat the same for other ROI - such as NAWM masks

## If you want to harmonise, then proceed to Step 4.1 or ignore the step altogether and proceed to 5
