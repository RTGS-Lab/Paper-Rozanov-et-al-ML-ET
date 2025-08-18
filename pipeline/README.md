# Data Processing Pipeline

This directory contains the scripts and notebooks for the data processing pipeline, which prepares the data for model training.

## Pipeline Stages

The pipeline is divided into several stages, each handled by a specific notebook or script:

- **1. ERA_Pipeline.ipynb**: This notebook processes the ERA data. It includes steps for downloading, cleaning, and transforming the raw ERA data into a suitable format.

- **2. MODIS_Pipeline.ipynb**: This notebook handles the processing of MODIS data. It includes similar steps for downloading, cleaning, and transforming the MODIS data.

- **3. DataStacking.ipynb**: This notebook is responsible for stacking the processed ERA and MODIS data together. It aligns the datasets in time and space to create a unified dataset for model input.

## Scripts

- **pipeline.py**: This is the main script for running the entire data processing pipeline in an automated fashion. It loads ERA5 and MODIS data, computes features, and predicts evapotranspiration (ET) using a pre-trained model.

- **driveDerive.py**: This script transfers data from a specified Google Drive folder to the local filesystem.

- **shedule_day.sh**: This shell script schedules the execution of the pipeline.

## Supporting Files

- **files2derive.txt**: A list of files that are used as input for the `driveDerive.py` script.

- **remaining_files.txt**: A list of files that are yet to be processed by the pipeline.

## Source Code

The `src` directory contains Python scripts with utility functions and configurations for the pipeline:

- **src/config.py**: Contains a list of feature names to ensure dataframes have the correct columns.
- **src/PM_eq.py**: Implements the Penman-Monteith equation for calculating reference evapotranspiration.
- **src/utils_era.py**: Contains utility functions for fetching and processing ERA5 data from Google Earth Engine.
- **src/utils_modis.py**: Contains utility functions for fetching and processing MODIS data from Google Earth Engine.
- **src/utils.py**: Contains general utility functions used throughout the pipeline, including feature computation and interpolation.