# Model Training and Validation Notebooks

This directory contains the Jupyter notebooks and source code for training, validating, and testing the machine learning model for evapotranspiration estimation.

## Notebooks

- **1. Final_Model_Trainging_and_Validation.ipynb**: This notebook details the process of training the final machine learning model. It includes steps for data loading, preprocessing, model training, and validation against the training dataset.

- **2. Model_Mesonet_Testing.ipynb**: This notebook is used for testing the trained model against data from the Mesonet network. It loads the trained model and evaluates its performance on the Mesonet dataset.

## Source Code

The `src` directory contains Python scripts that are used by the notebooks for various tasks:

- **src/PM_eq.py**: Implements the Penman-Monteith equation for calculating reference evapotranspiration. It supports calculations based on both ERA5 and ground-based data.
- **src/train.py**: Contains functions for training and evaluating various machine learning models, including LightGBM, XGBoost, Random Forest, CatBoost, and an Artificial Neural Network (ANN).
- **src/train_ann.py**: Defines the architecture and training procedures for the Artificial Neural Network (ANN) model, including data loading, scaling, and model checkpointing.