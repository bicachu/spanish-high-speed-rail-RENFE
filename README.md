# spanish-high-speed-rail-RENFE (Price Prediction)
__Objective:__ Build machine learning models that predice the price of a RENFE train ticket

The renfe dataset (input data) can be downloaded from Kaggle here: https://www.kaggle.com/thegurusteam/spanish-high-speed-rail-system-ticket-pricing. It consists of large dataset of 7+ million ticket prices that were scraped from the official Renfe website between April 2019 - August 2019 


### Reproduceability Steps
The following steps must be performed in order to properly reproduce and generate the correct model results

__Feature Engineering__

run _feature_engineering.py_ to import dataset, create all featured engineered, 
and clean data (handle missing and incorrect data)

__Model Preparation & Train/Test Splits__

_model_preparation.py_ is used to load in feature engineered and cleaned dataset

__Models__

run corresponding model notebooks used:
* Linear Regression
* Random Forest
* Gradient Boosting (GB Regressor, XGBoost, CatBoost)
* Neural Network 

### Additional Files in Repository

_datasets_ = directory that includes csv file of all holidays in Spain used for create days to/from holiday features
