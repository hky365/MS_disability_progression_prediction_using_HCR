# Radiomics Pipeline for Predicting Disability Progression in MS

### Overview
This repository contains scripts and pipelines for predicting disability progression in people with Multiple Sclerosis (PwMS) using machine learning (ML) and radiomics features derived from **T2-weighted FLAIR MRI**. This code was used for our study the details of which are given below.

### Title:
Leveraging Hand-Crafted Radiomics on Multicenter FLAIR MRI for Predicting Disability Progression in People with Multiple Sclerosis

### Authors: 
Hamza Khan (1) (2) (3), Henry C Woodruff (3) (4) and Diana L. Giraldo (5) (6), Lorin Werthen-Brabants (7), Shruti Atul Mali (3)(4), Sina Amirrajab (3), Edward De Brouwer (8), Veronica Popescu (1) (9), Bart Van Wijmeersch (1) (9), Oliver Gerlach (10) (11), Jan Sijbers (5) (6), Liesbet M. Peeters (1) (2) (9) and Philippe Lambin (3) (4).
### Affiliations: 
(1) University MS Center, Biomedical Research Institute (BIOMED), Hasselt University, Agoralaan Building C, 3590 Diepenbeek, Belgium.
(2) Data Science Institute (DSI), Hasselt University, Agoralaan Building D, 3590, Diepenbeek, Belgium.
(3) The D-Lab, Department of Precision Medicine, GROW - Research Institute for Oncology and Reproduction, Maastricht University, Maastricht, Netherlands.
(4) Department of Radiology and Nuclear Imaging, GROW –  Research Institute for Oncology and Developmental Biology, Maastricht University Medical Centre, Maastricht, Netherlands. 
(5) imec-Vision Lab, University of Antwerp, Antwerp, Belgium.
(6) 𝜇NEURO Research Centre of Excellence, University of Antwerp, Antwerp, Belgium.
(7) SUMO Group, IDLab, Ghent University - imec, Ghent, Belgium. 
(8) ESAT-STADIUS, KU Leuven, Belgium.
(9) Noorderhart, Rehabilitation and MS Center, Pelt, Belgium.
(10) Academic MS Center Zuyderland, Department of Neurology, Zuyderland Medical Center, Sittard-Geleen, The Netherlands.
(11) School for Mental Health and Neuroscience, Maastricht University, Maastricht, The Netherlands.

### Abstract
Multiple sclerosis (MS) is a chronic autoimmune disease of the central nervous system that results in varying degrees of functional impairment. Conventional tools, such as the Expanded Disability Status Scale (EDSS), lack sensitivity to subtle changes in disease progression. Radiomics offers a quantitative imaging approach to address this limitation. This study used machine learning (ML) and radiomics features derived from T2-weighted Fluid-Attenuated Inversion Recovery (FLAIR) magnetic resonance images (MRI) to predict disability progression in people with MS (PwMS).
A retrospective analysis was performed on real-world data from 247 PwMS across two centers. Disability progression was defined using EDSS changes over two years. FLAIR MRIs were preprocessed using bias-field correction, intensity normalisation, and super-resolution reconstruction for low-resolution images. White matter lesions (WML) were segmented using the Lesion Segmentation Toolbox (LST), and MRI tissue segmentation was performed using sequence Adaptive Multimodal SEGmentation. Radiomics features from WML and normal-appearing white matter (NAWM) were extracted using PyRadiomics, harmonised with Longitudinal ComBat, and reduced via Spearman correlation and recursive feature elimination. Elastic Net, Balanced Random Forest (BRFC), and Light Gradient-Boosting Machine (LGBM) models were evaluated on validation data and subsequently tested on unseen data.
The LGBM model with harmonised baseline features achieved the best test performance (PR AUC=0.20; ROC AUC=0.64). Key predictive features included GLCM_maximum probability (WML), GLDM_dependence non uniformity (NAWM), and left lateral ventrical volume. Short-term changes (delta radiomics) showed limited predictive power (PR AUC=0.11; ROC AUC=0.69).
These findings support the utility of baseline and delta radiomics and ML for predicting disability progression in PwMS. Future studies should validate these findings with larger datasets, more balanced classes, and additional imaging modalities.

### Practical Information about the repository
This repository includes the following steps:
### Repository Contents
| **Script**                          | **Description**                                      |
|-------------------------------------|------------------------------------------------------|
| `1 - Skull_Stripping_mgz_nii.py`    | Performs skull stripping on MRI data                |
| `2 - Intensity_Normalisation.py`    | Normalizes intensity across MR images               |
| `3 - Masks Extraction (No WML).py`  | Extracts masks for segmentation (excluding WML)     |
| `4 - Radiomics_Features_Extraction.py` | Extracts radiomics features using PyRadiomics       |
| `5 - Splitting_Data_Frames.py`      | Splits data into appropriate subsets                |
| `6 - Feature_Selection.py`          | Reduces features using correlation and RFE          |
| `7 - Model_training.py`             | Trains ML models (Elastic Net, BRFC, LGBM)          |
| `8 - Delta_dataframe.py`            | Generates delta radiomics features                  |

---

For steps preceding the Skull_Stripping_mgz_nii.py, kindly refer to https://github.com/diagiraldo/MS_MRI_processing/tree/main

For correspondence, kindly reach out to Hamza Khan (https://github.com/hky365) & Diana Giraldo (https://github.com/diagiraldo)
