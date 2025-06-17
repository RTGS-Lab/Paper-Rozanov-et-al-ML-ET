<h1 align="center"> Knowledge-Guided Tree-Based Models for Evapotranspiration Upscaling in the U.S. Midwest </h1>
<div align="center"> This repository contains the code that accompanies the paper<br><b>Knowledge-Guided Tree-Based Models for Evapotranspiration Upscaling in the U.S. Midwest.</b><br><a href='https://scholar.google.com/citations?user=DyM0AjAAAAAJ&hl=en'>Aleksei Rozanov</a>, 
    Samikshya Subedi, Vasudha Sharma, 
    <a href='https://scholar.google.com/citations?user=O7xJ4mcAAAAJ&hl=en&oi=ao'>Bryan Runck</a><br><mark>Link</mark></div>

<h2>Abstract</h2>
Evapotranspiration (ET) plays a critical role in the land-atmosphere interactions, yet its accurate quantification across various spatiotemporal scales remains a challenge. In situ measurement approaches, like eddy covariance (EC) or weather-based ET estimation, allow for measuring ET at a single location. Agricultural applications require estimates for each management zone over broad areas, making it infeasible to deploy sensing systems at each location. This study integrates tree-based and knowledge-guided machine learning (ML) techniques with multispectral remote sensing data, griddled meteorology and EC data to upscale ET across the Midwest United States. We compare four tree-based models - Random Forest, CatBoost, XGBoost, LightGBM - and a simple feed-forward artificial neural network in combination with features engineered using knowledge-guided ML principles. Models were trained and tested on EC towers located in the Midwest of the United States using k-fold cross validation with k=10 and site-year, biome stratified train-test split to avoid data leakage. Results show that LightGBM with knowledge-guided features outperformed other methods with an R<sup>2</sup>=0.855, MSE=15.155 W·m<sup>-2</sup> and MAE = 9.166 W·m<sup>-2</sup> according to grouped k-fold validation (k=5). Feature importance analysis shows that knowledge-guided features were most important for predicting evapotranspiration.  Using the best performing model, we provide a data product at 500 m spatial and one-day temporal resolution for gridded ET for the period of 2019-2024.

<h2>Content</h2>
<code>
ET_LCCMR/
    ├── LICENSE
    ├── README.md
    ├── requirements.txt
    ├── requirements_local.txt
    ├── .gitignore
    ├── paper/
    ├── models/
        └── ligthgbm_model.txt
    ├── preprocessing
        └── 1. Carbon_Data.ipynb
        └── 2. Site_Selection.ipynb
        └── 3. Stack_Data.ipynb
        └── 4. Mesonet_ValidationData.ipynb
    ├── notebooks/
        └── src/
            └── PM_eq.py
            └── train_ann.py
            └── train.py
        └── 1. Final_Model_Training_and_Validation.ipynb
        └── 2. Model_Mesonet_Testing.ipynb
    ├── pipeline/
        └── src/
            └── config.py
            └── PM_eq.py
            └── utils_era.py
            └── utils_modis.py
        └── driveDerive.py
        └── 1. ERA_Pipeline.ipynb
        └── 2. MODIS_Pipeline.ipynb
        └── 3. DataStacking.ipynb

</code>
