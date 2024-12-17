# Mask Extraction Script (No WML)
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script extracts specific masks (excluding WML) from preprocessed MRI directories.
# It searches subject and session directories and processes available masks efficiently.

import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pandas as pd


#base_path = r'...\path_to_folder_with_SAMSEG_masks'


# ## Base_PATH



base_dir = base_path
# Define a Function to Extract the Masks
def extract_masks_from_samseg(samseg_data, lesion_data):
    # Masks for GM, CSF, and NAWM
    gm_labels = [3, 42, 8, 47]
    csf_labels = [4, 43, 5, 44, 14, 15, 24]
    nawm_labels = [2, 41, 46, 7]
    thalamus_labels = [10, 49]

    gm_mask = np.isin(samseg_data, gm_labels)
    csf_mask = np.isin(samseg_data, csf_labels)
    nawm_mask = np.isin(samseg_data, nawm_labels)
    thalamus_mask = np.isin(samseg_data, thalamus_labels)

    # Refine the NAWM mask by subtracting the lesion mask
    nawm_mask = np.logical_and(nawm_mask, np.logical_not(lesion_data))

    return gm_mask.astype(np.int16), csf_mask.astype(np.int16), nawm_mask.astype(np.int16), thalamus_mask.astype(np.int16)


# Get the list of subjects
subjects_list = os.listdir(base_dir)

# Lists to store skipped subjects and sessions
skipped = []

# Iterate through each subject and session with tqdm progress bar
for subject in tqdm(subjects_list, desc="Processing Subjects"):
    subject_dir = os.path.join(base_dir, subject)
    
    for session in os.listdir(subject_dir):
        session_dir = os.path.join(subject_dir, session, "anat")
    
        # Dynamically check for directories with 'samseg' and 'LST'
        samseg_dir = next((d for d in os.listdir(session_dir) if 'samseg' in d), None)
        lst_dir = next((d for d in os.listdir(session_dir) if 'LST' in d), None)

        if samseg_dir and lst_dir:
            samseg_output_path = os.path.join(session_dir, samseg_dir, "seg.mgz")
            lesion_mask_path = os.path.join(session_dir, lst_dir, "ples_lpa.nii.gz")

            if os.path.exists(samseg_output_path) and os.path.exists(lesion_mask_path):
                samseg_img = nib.load(samseg_output_path)
                samseg_data = samseg_img.get_fdata()

                lesion_data = nib.load(lesion_mask_path).get_fdata()

                gm_mask, csf_mask, nawm_mask, thalamus_mask = extract_masks_from_samseg(samseg_data, lesion_data)

                # Save the masks
                nib.save(nib.Nifti1Image(gm_mask, samseg_img.affine), os.path.join(session_dir, 'gm_mask.nii.gz'))
                nib.save(nib.Nifti1Image(csf_mask, samseg_img.affine), os.path.join(session_dir, 'csf_mask.nii.gz'))
                nib.save(nib.Nifti1Image(nawm_mask, samseg_img.affine), os.path.join(session_dir, 'nawm_mask.nii.gz'))
                nib.save(nib.Nifti1Image(thalamus_mask, samseg_img.affine), os.path.join(session_dir, 'thalamus_mask.nii.gz'))
            else:
                print(f"Skipped mask extraction for {subject} - Session {session} due to missing files.")
                skipped.append((subject, session))

# Print the skipped subjects and sessions
print("\nSkipped subjects and sessions:")
for s, sess in skipped:
    print(f"Subject: {s}, Session: {sess}")


# Define mask names and suffix
masks_to_binarize = ['nawm_mask.nii.gz', 'gm_mask.nii.gz', 'thalamus_mask.nii.gz']
lst_mask = 'ples_lpa.nii.gz'
wml_mask = 'wml_mask.nii.gz'
suffix = '_binarized.nii.gz'

# List to store any errors
error_log = []

def binarize_and_save_mask(file_path, output_path):
    try:
        # Load the mask
        mask_img = nib.load(file_path)
        mask_data = mask_img.get_fdata()

        # Binarize the mask
        binary_mask = (mask_data > 0).astype(np.uint8)

        # Save the binarized mask
        binarized_img = nib.Nifti1Image(binary_mask, mask_img.affine)
        nib.save(binarized_img, output_path)
        print(f"Binarized mask saved at: {output_path}")

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        error_log.append({'file_path': file_path, 'error': str(e)})

# Loop over patients and sessions with tqdm for progress
for patient in tqdm(os.listdir(pelt_path), desc="Processing Patients"):
    patient_path = os.path.join(pelt_path, patient)
    if os.path.isdir(patient_path):
        for session in os.listdir(patient_path):
            session_path = os.path.join(patient_path, session, 'anat')
            
            # Check if 'anat' folder exists
            if os.path.isdir(session_path):
                # Binarize each specified mask in the 'anat' folder
                for mask_name in masks_to_binarize:
                    mask_path = os.path.join(session_path, mask_name)
                    if os.path.exists(mask_path):
                        output_path = mask_path.replace('.nii.gz', suffix)
                        print(f"Processing mask: {mask_path}")
                        binarize_and_save_mask(mask_path, output_path)
                
                # Check for LST mask in the LST folder
                lst_folder = os.path.join(session_path, 'LST')
                lst_mask_path = os.path.join(lst_folder, lst_mask)
                
                if os.path.exists(lst_mask_path):
                    output_path = lst_mask_path.replace('.nii.gz', suffix)
                    print(f"Processing LST mask: {lst_mask_path}")
                    binarize_and_save_mask(lst_mask_path, output_path)
                else:
                    # Fallback to WML mask in 'anat' if LST folder is not present or 'ples_lpa.nii.gz' is missing
                    wml_path = os.path.join(session_path, wml_mask)
                    if os.path.exists(wml_path):
                        output_path = wml_path.replace('.nii.gz', suffix)
                        print(f"Processing fallback WML mask: {wml_path}")
                        binarize_and_save_mask(wml_path, output_path)

# Convert the error log to a DataFrame and save if there are any errors
if error_log:
    error_df = pd.DataFrame(error_log)
    error_df.to_csv(os.path.join(pelt_path, 'binarization_errors.csv'), index=False)
    print("Error log saved to 'binarization_errors.csv'")