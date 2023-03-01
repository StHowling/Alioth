# %%
from cProfile import label
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams["font.family"] = "Arial"
# %%
def exchange_row(df, i, j):
    tmp = df.iloc[i].copy()
    df.iloc[i] = df.iloc[j]
    df.iloc[j] = tmp

# %%
# best_df = pd.read_csv("best_pred_loss.csv")
cluster_df = pd.read_csv("cluster_CPI.csv")
# worst_df = pd.read_csv("worst_cluster_CPI.csv")
# %%
thresh = 1.05
# b_df1 = best_df[best_df["threshold"] == thresh].copy()
c_df1 = cluster_df[cluster_df["threshold"] == thresh].copy()
# w_df1 = worst_df[worst_df["threshold"] == thresh].copy()
app = [
"cassandra", 
"etcd", 
"hbase", 
"milc", 
"kafka", 
"mongoDB", 
"rabbitmq", 
"redis"
]
APP = [
"Cassandra",
"Etcd",
"HBase",
"HPC",
"Kafka",
"MongoDB",
"RabbitMQ",
"Redis", 
]
c_01 = []
for i in app:
    c_01.append(c_df1[c_df1["app"] == i]["01_acc"].values)
c_01 = np.squeeze(np.array(c_01))
# equa_c = [0.5531, 0.5339, 0.5595, 0.5672, 0.5732, 0.54, 0.6537, 0.5122]
equa_c = [0.5531, 0.5339, 0.5729, 0.5732, 0.5672, 0.5595, 0.6537, 0.5122]
# %%
x = np.arange(len(c_df1))
bar_width = 0.3
plt.figure(figsize=(10, 2))
# plt.bar(x - bar_width, b_df1["mae"], bar_width, label="Best-Possible", color ="#66c2a5")
# plt.bar(x, c_df1["mae"], bar_width, tick_label=app, label="Best-Effort", alpha=0.8, color= "#4169E1")
# plt.bar(x + bar_width, w_df1["mae"], bar_width, label="Least-Effort", color = "#F0E68C")
# plt.bar(x - 0.5 *bar_width, b_df1["01_acc"], bar_width, label="CPI", color ="#66c2a5")
# plt.bar(x + 0.5 * bar_width, equa_c, bar_width, label="Equation", alpha=0.8, color= "#4169E1")
plt.plot(x, c_01, label="CPI")
plt.plot(x, equa_c, label="Formula")
plt.xticks(x, APP)
# plt.bar(x + bar_width, w_df1["mae"], bar_width, label="Least-Effort", color = "#F0E68C")
plt.legend()
plt.xlabel("Applications")
plt.ylabel("Accuracy")
plt.grid(linestyle = '-.')
# plt.show()
os.chdir(r"plot")
plt.savefig("CPI_Formula.pdf", bbox_inches='tight')
# %%

# plt.plot(x, b_df1["01_acc"], label="best")
# plt.plot(x, c_df1["01_acc"], label="cluster")
# plt.plot(x, w_df1["01_acc"], label="worst")
# plt.xticks(x, b_df1["app"])
# plt.legend()
# plt.show()

# %%
