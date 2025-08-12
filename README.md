<h1 align="center"> Knowledge-Guided Tree-Based Models for Evapotranspiration Upscaling in the U.S. Midwest </h1>
<div align="center"> This repository contains the code that accompanies the paper<br><b>Knowledge-Guided Tree-Based Models for Evapotranspiration Upscaling in the U.S. Midwest.</b><br><a href='https://scholar.google.com/citations?user=DyM0AjAAAAAJ&hl=en'>Aleksei Rozanov</a>, 
    Samikshya Subedi, Vasudha Sharma, 
    <a href='https://scholar.google.com/citations?user=O7xJ4mcAAAAJ&hl=en&oi=ao'>Bryan Runck</a><br><mark>Link</mark></div>

<h2>Abstract</h2>
Evapotranspiration (ET) plays a critical role in the land-atmosphere interactions, yet its accurate quantification across various spatiotemporal scales remains a challenge. In situ measurement approaches, like eddy covariance (EC) or weather station-based ET estimation, allow for measuring ET at a single location. Agricultural uses of ET require estimates for each field over broad areas, making it infeasible to deploy sensing systems at each location. This study integrates tree-based and knowledge-guided machine learning (ML) techniques with multispectral remote sensing data, griddled meteorology and EC data to upscale ET across the Midwest United States. We compare four tree-based models - Random Forest, CatBoost, XGBoost, LightGBM - and a simple feed-forward artificial neural network in combination with features engineered using knowledge-guided ML principles. Models were trained and tested on EC towers located in the Midwest of the United States using k-fold cross validation with k=5 and site-year, biome stratified train-test split to avoid data leakage. Results show that LightGBM with knowledge-guided features outperformed other methods with an R2=0.86, MSE=14.99 W·m<sup>-2</sup> and MAE = 8.82 W·m<sup>-2</sup> according to grouped k-fold validation (k=5). Feature importance analysis shows that knowledge-guided features were most important for predicting evapotranspiration. Using the best performing model, we provide a data product at 500 m spatial and one-day temporal resolution for gridded ET for the period of 2019-2024. Intercomparison between the new gridded product and state-level weather station-based ET estimates show best-in-class correspondence. 

<h2>Content</h2>
<code>ET_LCCMR/
    ├─ LICENSE
    ├─ README.md
    ├─ requirements.txt              # Core deps for MSI usage
    ├─ requirements_local.txt        # Extras for local/dev use
    ├─ .gitignore
    ├─ paper/                        # Drafts, manuscript assets
    ├─ fig/                          # Exported figures
    ├─ models/
    │  └─ ligthgbm_model.txt         
    ├─ preprocessing/                # One-off data prep notebooks (site/stack/mesonet)
    │  ├─ 1. Carbon_Data.ipynb
    │  ├─ 2. Site_Selection.ipynb
    │  ├─ 3. Stack_Data.ipynb
    │  └─ 4. Mesonet_ValidationData.ipynb
    ├─ notebooks/
    │  ├─ src/
    │  │  ├─ PM_eq.py                # PM helpers (LE↔ET conversion, etc.)
    │  │  ├─ train_ann.py            # Simple ANN training
    │  │  └─ train.py                # Tree-models training / utils
    │  ├─ 1. Final_Model_Training_and_Validation.ipynb
    │  └─ 2. Model_Mesonet_Testing.ipynb
    └─ pipeline/                     # Data pipelines (ERA, MODIS) + utilities
       ├─ src/
       │  ├─ config.py               # Centralized config
       │  ├─ PM_eq.py                # PM helpers (duplicated with notebooks/src/)
       │  ├─ utils_era.py            # ERA5-Land acquisition and transforms
       │  └─ utils_modis.py          # MODIS acquisition and transforms
       ├─ driveDerive.py             
       ├─ 1. ERA_Pipeline.ipynb
       ├─ 2. MODIS_Pipeline.ipynb
       └─ 3. DataStacking.ipynb
</code>

<h2>Getting started</h2>
To replicate the analysis presented in this repository, follow these steps:

### 1. Clone the Repository
First, clone the repository to your local machine:
```
git clone https://github.com/RTGS-Lab/ET_LCCMR.git
cd ET_LCCMR
```

### 2. Set Up the Python Environment
It is highly recommended to use a virtual environment to manage dependencies.
```
  python -m venv .venv
  source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
  pip install -r requirements_local.txt # Or requirements.txt to run on MSI
```
### 3. Use the notebooks
Navigate to the `preprocessing/` directory and run the Jupyter notebooks in the specified order to prepare the data for model training:

`  1. Carbon_Data.ipynb
 2. Site_Selection.ipynb
 3. Stack_Data.ipynb
 4. Mesonet_Validation_Data.ipynb`





