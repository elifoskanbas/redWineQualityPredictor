# Red Wine Quality Predictor – Regression Model
This project uses the Red Wine Quality Dataset (Cortez et al., 2009) 
from Kaggle to build a regression model that predicts the quality of red wine based on its physicochemical properties.

The goal is to predict wine quality scores (ranging from 0 to 10) using features such as acidity, pH, alcohol content, and others.

## Project Overview
The notebook evaluates three different regression models to predict wine quality:

**1.Linear Regression** 

**2. Support Vector Regressor (SVR)**  with hyperparameter tuning using Grid Search

**3. Random Forest Regressor**  with hyperparameter tuning using 
Grid Search

Among these, the Random Forest model achieved the best performance.

## Dataset
**Source:** Kaggle - Red Wine Quality Dataset

**File:** winequality-red.csv

**Number of Features:** 11

**Target Variable:** quality

**Sample Data (first 5 rows)**

| fixed acidity | volatile acidity | citric acid | residual sugar | chlorides | ... | alcohol | quality |
|----------------|------------------|--------------|----------------|------------|-----|----------|----------|
| 7.4 | 0.70 | 0.00 | 1.9 | 0.076 | ... | 9.4 | 5 |
| 7.8 | 0.88 | 0.00 | 2.6 | 0.098 | ... | 9.8 | 5 |
| 7.8 | 0.76 | 0.04 | 2.3 | 0.092 | ... | 9.8 | 5 |
| 11.2 | 0.28 | 0.56 | 1.9 | 0.075 | ... | 9.8 | 6 |
| 7.4 | 0.70 | 0.00 | 1.9 | 0.076 | ... | 9.4 | 5 |


## Setup and Execution
**Requirements**

This project is designed for Google Colab.

The following libraries are required:
```bash
pandas
scikit-learn
zipfile
os
kaggle
```

**Step 1:** Configure Kaggle API

Upload your kaggle.json API key file to your Colab environment:
```bash
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Step 2:** Download and Extract Dataset
```bash
!kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009

import zipfile, os, pandas as pd

zip_filename = 'red-wine-quality-cortez-et-al-2009.zip'
extract_dir = './wine_data'

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

df = pd.read_csv(os.path.join(extract_dir, 'winequality-red.csv'))
```

**Step 3:** Data Preprocessing

- **Target variable:** `quality`  
- **Feature standardization:** `StandardScaler`  
- **Train-test split:** 80% training, 20% testing  


**Step 4:** Model Training and Evaluation

***Linear Regression***
```bash
Mean Squared Error: 0.3900  
R² Score: 0.4031
```
***Support Vector Regressor (with Grid Search)***
```bash
Best Parameters:
{'C': 1, 'epsilon': 0.2, 'gamma': 0.1}

Performance:
MSE: 0.3443  
R²: 0.4731
```
***Random Forest Regressor (with Grid Search)***
```bash
Best Parameters:
{'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}

Performance:
MSE: 0.3067  
R²: 0.5307
```

**Model Comparison**

| Model | Mean Squared Error | R² Score |
|--------|--------------------|----------|
| Linear Regression | 0.3900 | 0.4031 |
| SVR (Tuned) | 0.3443 | 0.4731 |
| Random Forest (Tuned) | 0.3067 | 0.5307 |


***Best Model: Random Forest Regressor***

## Conclusions

The physicochemical features of wine can moderately predict wine quality.

The Random Forest Regressor achieved the best performance among the tested models.

Future improvements could include:

- Feature engineering and selection
- Larger and more diverse datasets
- Model explainability using SHAP or LIME
