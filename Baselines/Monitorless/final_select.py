# %%
import os
import logging
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
# %%
# DATADIR = r"E:\research\cloudVM\code\data\interference_data\monitorless_output\raw"
# OUTPUT_DIR = r"E:\research\cloudVM\code\data\interference_data\monitorless_output\select"
DATADIR = r"../data/monitorless_output/raw"
OUTPUT_DIR = r"../data/monitorless_output/select"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

logfile = "./select.log"
logger = logging.getLogger("select features logger")
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
top_features = {}
for i in csv_file_list:
    filename = os.path.basename(i)
    app = filename.split(".")[0]
    logger.info("Dealing with {}".format(app))

    data = pd.read_csv(i)
    data_keys = list(data.columns)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    regr = RandomForestRegressor(random_state=0, n_jobs=-1)
    regr.fit(X, y)
    fl = regr.feature_importances_
    fl_sort = np.argsort(fl)[::-1][:30]

    df_values = data.iloc[:, fl_sort]
    sort_features = list(df_values.columns)
    logger.info("first select top features: {}".format(sort_features))
    # df_final = pd.concat([df_values, data.iloc[:, -1]], axis=1)
    # df_final.to_csv(os.path.join(OUTPUT_DIR, app + ".csv"), index=False)
    # Combining features
    for j in range(len(sort_features)):
        for k in range(j + 1, len(sort_features)):
            tmp1 = df_values.iloc[:, j]
            tmp2 = df_values.iloc[:, k]
            name = tmp1.name + "x" + tmp2.name
            tmp = tmp1 * tmp2
            tmp.name = name
            df_values = pd.concat([df_values, tmp], axis=1)
    # Add time-dependent features
    for j in sort_features:
        tavg_list = []
        tlag_list = []
        for k in range(len(df_values)):
            start = k - 15
            if start < 0:
                start = 0
            tavg_list.append(df_values[j][start: k + 1].mean())
            tlag_list.append(df_values[j][start])
        tpd = pd.DataFrame(np.array([tavg_list, tlag_list]).T, columns=[j + "avg", j + "lag"])
        df_values = pd.concat([df_values, tpd], axis=1)
    df_final = pd.concat([df_values, data.iloc[:, -1]], axis=1)

    # Final pred
    start_time = time.time()
    X = df_values.values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regr = RandomForestRegressor(random_state=0, n_jobs=-1)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    mse_loss = mean_squared_error(y_test, y_pred)
    mae_loss = mean_absolute_error(y_test, y_pred)
    end_time = time.time()
    logger.info("Final MSE loss: {}, MAE loss: {}, time: {}".format(mse_loss, mae_loss, end_time - start_time))

    # Final select
    fl = regr.feature_importances_
    fl_sort = np.argsort(fl)[::-1][:30]
    final_values = df_values.iloc[:, fl_sort]
    final_features = list(final_values.columns)
    top_features[app] = final_features
    logger.info("final select top features: {}".format(final_features))
    df_final = pd.concat([final_values, data.iloc[:, -1]], axis=1)
    df_final.to_csv(os.path.join(OUTPUT_DIR, app + ".csv"), index=False)

# %%
