# %%
import os
import logging
import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# %%
DATADIR = r"E:\research\cloudVM\code\data\interference_data\practical"
# DATADIR = r"../data/pratical"

logfile = "./practical.log"
logger = logging.getLogger("practical method logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# %%
def find_all_csv(relative_path, file_list, file_name="-1.csv"):
    for i in os.listdir(relative_path):
        file_path = os.path.join(relative_path, i)
        if os.path.isdir(file_path):
            find_all_csv(file_path, file_list, file_name)
        else:
            if i.endswith(file_name):
                file_list.append(file_path)

# %%
csv_file_list = []
find_all_csv(DATADIR, csv_file_list, ".csv")
# %%
for i in csv_file_list:
    filename = os.path.basename(i)
    app = filename.split(".")[0]
    df = pd.read_csv(i)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    start_time = time.time()
    X_select = SelectKBest(f_regression, k=30).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size=0.2, random_state=0)
    regr = BaggingRegressor(DecisionTreeRegressor(), random_state=0)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    end_time = time.time()
    mse_loss = mean_squared_error(y_test, y_pred)
    mae_loss = mean_absolute_error(y_test, y_pred)
    logger.info("{} MSE loss: {}, MAE loss: {}, time: {}".format(app, mse_loss, mae_loss, end_time - start_time))
