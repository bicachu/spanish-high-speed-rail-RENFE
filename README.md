# spanish-high-speed-rail-RENFE

The renfe dataset (input data) can be downloaded from Kaggle here: https://www.kaggle.com/thegurusteam/spanish-high-speed-rail-system-ticket-pricing. It consists of large dataset of 7+ million ticket prices that were scraped from the official Renfe website between April 2019 - August 2019 


### Reproduceability Steps
The following steps must be performed in order to properly reproduce and generate the correct model results

__Feature Engineering__

run feature_engineering.py to import dataset, create all featured engineered, 
and clean data (handle missing and incorrect data)

__Model Preparation & Train/Test Splits__

model_preparation.py is used to load in feature engineered and cleaned dataset

__Models__

_run corresponding model notebooks used:_
* Linear Regression
* Random Forest
* Gradient Boosting (GB Regressor, XGBoost, CatBoost)
* Neural Network 

### Additional Files in Repository

datasets = directory that includes csv file of all holidays in Spain used for create days to/from holiday features
