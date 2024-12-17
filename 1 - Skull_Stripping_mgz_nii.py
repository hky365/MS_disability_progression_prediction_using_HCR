# Skull Stripping Script
# Author: Hamza Khan, UHasselt/UMaastricht
# This script processes MRI images for skull stripping by converting files between formats,
# applying masks, and organising output.

import os
import nibabel as nib
import numpy as np

# Function to convert MGZ to NIfTI
def convert_mgz_to_nifti(input_path, output_path):
    """
    Converts an MGZ file to a NIfTI format.

    Parameters:
    input_path (str): Path to the input MGZ file.
    output_path (str): Path to save the converted NIfTI file.
    """
    try:
        mgz_image = nib.load(input_path)
        nib.save(mgz_image, output_path)
        print(f"Converted MGZ to NIfTI: {output_path}")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

# Function to load and apply a brain mask to an image
def apply_brain_mask(image_path, mask_path, output_path):
    """
    Applies a brain mask to a given image and saves the result.

    Parameters:
    image_path (str): Path to the input image (e.g., .nii or .mgz file).
    mask_path (str): Path to the brain mask.
    output_path (str): Path to save the masked image.
    """
    try:
        # Load the image and the mask
        image = nib.load(image_path)
        mask = nib.load(mask_path)

        # Ensure the dimensions match
        if image.shape != mask.shape:
            raise ValueError(f"Shape mismatch: image {image.shape} and mask {mask.shape}")

        # Apply the mask
        image_data = image.get_fdata()
        mask_data = mask.get_fdata()
        masked_data = np.where(mask_data > 0, image_data, 0)

        # Save the masked image
        masked_image = nib.Nifti1Image(masked_data, image.affine, image.header)
        nib.save(masked_image, output_path)
        print(f"Masked image saved to: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Directory setup
input_dir = "path/to/input"  # Replace with the input directory
output_dir = "path/to/output"  # Replace with the output directory
mask_dir = "path/to/masks"  # Replace with the mask directory

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through input directory and process files
for subject in os.listdir(input_dir):
    subject_path = os.path.join(input_dir, subject)
    if not os.path.isdir(subject_path):
        continue

    for session in os.listdir(subject_path):
        session_path = os.path.join(subject_path, session)
        if not os.path.isdir(session_path):
            continue

        # Define file paths
        mgz_path = os.path.join(session_path, "image.mgz")  # Input MGZ file
        nifti_path = os.path.join(session_path, "image.nii.gz")  # Converted NIfTI file
        mask_path = os.path.join(mask_dir, subject, session, "mask.nii.gz")  # Brain mask
        output_path = os.path.join(output_dir, subject, session, "masked_image.nii.gz")

        # Ensure the session's output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert MGZ to NIfTI if needed
        if os.path.exists(mgz_path):
            convert_mgz_to_nifti(mgz_path, nifti_path)

        # Apply the brain mask
        if os.path.exists(nifti_path) and os.path.exists(mask_path):
            apply_brain_mask(nifti_path, mask_path, output_path)
        else:
            print(f"Missing files for subject {subject}, session {session}.")

print("Processing complete.")
