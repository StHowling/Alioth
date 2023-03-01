# %%
import os
import logging
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
# import plotly.graph_objects as go
# import plotly.offline as pyo

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
fl_all = {}
for i in csv_file_list:
    filename = os.path.basename(i)
    app = filename.split(".")[0]
    if app == "total":
        continue
    # if app not in ["hbase", "cassandra"]:
        # continue
    df = pd.read_csv(i)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    regr = RandomForestRegressor(random_state=0, n_jobs=-1)
    regr.fit(X, y)
    keys = np.array(df.iloc[:, :-1].columns)
    fl = regr.feature_importances_
    fl_sort = np.argsort(fl)[::-1][:8]
    fl_label = keys[fl_sort]
    fl_score = fl[fl_sort]
    fl_all[app] = []
    fl_all[app].append(fl_label)
    fl_all[app].append(fl_score)

# %%
label = ['MBR', 'MBL', 'kbcached', 'IPC', 'system_time', 'idle', 'net_rd_packet']
label = [*label, label[0]]
cassandra_label = fl_all["cassandra"][0][[0, 1, 2, 6, 7]]
cassandra_score = fl_all["cassandra"][1][[0, 1, 2, 6, 7]]
cassandra_score_final = [0.6461558 , 0.67563752, 0.61294016, 0.6056449 , 0.60533717, 0.3, 0.3]
cassandra_score_final = [*cassandra_score_final, cassandra_score_final[0]]
hbase_label = fl_all["hbase"][0][[0, 1, 2, 4, 5]]
hbase_score = fl_all["hbase"][1][[0, 1, 2, 4, 5]]
hbase_score_final = [0.72513976, 0.62187549, 0.3, 0.52129709, 0.3, 0.51428043, 0.61258208]
hbase_score_final = [*hbase_score_final, hbase_score_final[0]]
# %%
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(10, 10))
plt.subplot(polar=True)
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(hbase_score_final))
plt.plot(label_loc, cassandra_score_final, label="Cassandra")
plt.plot(label_loc, hbase_score_final, label="HBase")
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=label, fontsize=30)
plt.legend(fontsize=30, loc="upper left")
# plt.show()
os.chdir(r"E:\research\cloudVM\papers\sigmetrics\plot")
plt.savefig("App_radar.pdf", bbox_inches='tight')

# %%
# fig = go.Figure(
#     data=[
#         go.Scatterpolar(r=cassandra_score_final, theta=label, fill='toself', name='Cassandra'),
#         go.Scatterpolar(r=hbase_score_final, theta=label, fill='toself', name='HBase')
#     ],
#     layout=go.Layout(
#         title=go.layout.Title(text='Application Variance'),
#         polar={'radialaxis': {'visible': True}},
#         showlegend=True
#     )
# )

# pyo.plot(fig)
# %%
