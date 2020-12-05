This code repo contains the following files:
- Data files
-- train.csv
-- test.csv
-- 2017_2018PublicHolidayWithoutSunday.csv
- Program sources
-- MScBDT5001IndividualProjectDP.ipynb
-- MScBDT5001IndividualProjectNN&XGBoost.ipynb
- Result files
-- train_data_prepared.pkl
-- best_model_file.h5
-- pima.pickle.dat
-- Submission.csv
- Readme file
-- readme.txt

Programming languages used.
- Python is used for the individual project.
- Coding is done in google colaboratory.
- The program sources are saved as MScBDT5001IndividualProjectDP.ipynb and MScBDT5001IndividualProjectNN&XGBoost.ipynb

MScBDT5001IndividualProjectDP.ipynb is for data preparation.
Data file required - train.csv and 2017_2018PublicHolidayWithoutSunday.csv
Result file - train_data_prepared.pkl

The following packages and libraries are required for data preparation:
- from datetime import datetime, timedelta, date
- from IPython.display import display
- from keras.backend import clear_session
- from keras.callbacks import EarlyStopping
- from keras.layers import Dense
- from keras.models import load_model, Sequential
- from keras.optimizers import Adam, RMSprop, SGD
- from pandas.plotting import scatter_matrix
- from sklearn.metrics import r2_score
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import MinMaxScaler
- from sys import exit
- from tensorflow.keras import activations
- import math
- import matplotlib.pyplot as plt
- import numpy as np
- import pandas as pd
- import pickle as pkl
- import seaborn as sns
- import statsmodels.api as sm
- import sys
- import tensorflow as tf
- import time

MScBDT5001IndividualProjectNN&XGBoost.ipynb is for modelling.
Data file required - train_data_prepared.pkl, 2017_2018PublicHolidayWithoutSunday.csv and test.csv
Result files - -- best_model_file.h5, pima.pickle.dat, Submission.csv

The following packages and libraries are only required for training:
- pip install bayesian-optimization
- from bayes_opt import BayesianOptimization

The following packages and libraries are required for training and prediction:
- from datetime import datetime, timedelta, date
- from IPython.display import display
- from keras.backend import clear_session
- from keras.callbacks import EarlyStopping
- from keras.layers import Dense
- from keras.models import load_model, Sequential
- from keras.optimizers import Adam, RMSprop, SGD
- from pandas.plotting import scatter_matrix
- from sklearn import tree
- from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
- from sklearn.linear_model import LinearRegression
- from sklearn.metrics import mean_squared_error, r2_score
- from sklearn.model_selection import train_test_split, cross_val_score
- from sklearn.pipeline import Pipeline
- from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
- from sys import exit
- from tensorflow.keras import activations
- import math
- import matplotlib.pyplot as plt
- import numpy as np
- import pandas as pd
- import pickle as pkl
- import seaborn as sns
- import statsmodels.api as sm
- import sys
- import tensorflow as tf
- import time
- import warnings
- import xgboost as xgb

Reproduce the result by the following steps:
1. Import library by running code block 1
2. Upload best_model_file.h5, pima.pickle.data, test.csv, 2017_2018PublicHolidayWithoutSunday.csv
3. Load neural network model by running code block 23
4. Load xgboost model by running code block 36
5. Reproduce my result (Submission.csv) by running code block 37 to 59
