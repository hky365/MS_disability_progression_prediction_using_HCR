# Intensity Normalization Script
# Author: Hamza Khan (UHasselt/UMaastricht)
# This script normalizes image intensities by performing histogram matching using a high-quality reference image.
# The target regions are NAWM and white matter lesions.

import os
import SimpleITK as sitk
from tqdm import tqdm

# Function for intensity normalization using histogram matching
def normalize_intensity(image_path, reference_image_path, output_path):
    """
    Performs histogram matching to normalize intensities of an image relative to a reference image.

    Parameters:
    image_path (str): Path to the input skull-stripped image to normalize.
    reference_image_path (str): Path to the high-quality reference image.
    output_path (str): Path to save the normalized image.
    """
    try:
        # Load the target and reference images
        target_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        reference_image = sitk.ReadImage(reference_image_path, sitk.sitkFloat32)

        # Perform histogram matching
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(255)
        matcher.SetNumberOfMatchPoints(50)
        matcher.ThresholdAtMeanIntensityOn()  # Avoid outliers

        # Match the intensity
        normalized_image = matcher.Execute(target_image, reference_image)

        # Save the output
        sitk.WriteImage(normalized_image, output_path)
        print(f"Normalized image saved to: {output_path}")
    except Exception as e:
        print(f"Error normalizing {image_path}: {e}")

# Function to collect skull-stripped images for processing
def collect_files(base_path, output_suffix="normalized_image_final.nii.gz"):
    """
    Traverses the directory structure to collect skull-stripped image paths.

    Parameters:
    base_path (str): Root directory containing subject/session subfolders.
    output_suffix (str): Suffix for the output normalized images.

    Returns:
    list: A list of tuples containing (input_path, output_path).
    """
    file_list = []
    for root, _, files in os.walk(base_path):
        if 'prettier_samseg' in root or 'prettier_LST' in root:
            continue

        stripped_image_path = os.path.join(root, "stripped_image_final.nii.gz")
        if os.path.exists(stripped_image_path):
            output_path = os.path.join(root, output_suffix)
            file_list.append((stripped_image_path, output_path))
    return file_list

# Directory paths (replace with your repository-style relative paths)
base_dirs = [
    "path/to/processed_MRI",
    "path/to/MS_MRI_2",
    "path/to/protocol_C",
    "path/to/processed_MRI_zuy",
    "path/to/processed_MRI_zuy_HR"
]
reference_image_path = "path/to/high_quality_reference_image.nii.gz"  # High-quality reference image

# Main processing loop
for base_path in base_dirs:
    print(f"Processing directory: {base_path}")
    file_list = collect_files(base_path)
    os.makedirs(os.path.dirname(reference_image_path), exist_ok=True)

    for image_path, output_path in tqdm(file_list, desc=f"Normalizing intensities in {os.path.basename(base_path)}"):
        normalize_intensity(image_path, reference_image_path, output_path)

print("Intensity normalization complete for all directories.")
