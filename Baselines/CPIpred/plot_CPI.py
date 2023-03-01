# %%
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
best_df = pd.read_csv("best_pred_loss.csv")
cluster_df = pd.read_csv("cluster_CPI.csv")
# worst_df = pd.read_csv("worst_cluster_CPI.csv")
# %%
thresh = 1.05
b_df1 = best_df[best_df["threshold"] == thresh].copy()
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
b_mae = []
c_mae = []
for i in app:
    b_mae.append(b_df1[b_df1["app"] == i]["mae"].values)
    c_mae.append(c_df1[c_df1["app"] == i]["mae"].values)
b_mae = np.squeeze(np.array(b_mae))
c_mae = np.squeeze(np.array(c_mae))

# %%
tick = np.arange(len(b_mae))
x = np.copy(tick).astype(float)
x[6] = 6.1
print(x)
bar_width = 0.3
plt.figure(figsize=(10, 4))
# plt.bar(x - bar_width, b_df1["mae"], bar_width, label="Best-Possible", color ="#66c2a5")
# plt.bar(x, c_df1["mae"], bar_width, tick_label=app, label="Best-Effort", alpha=0.8, color= "#4169E1")
# plt.bar(x + bar_width, w_df1["mae"], bar_width, label="Least-Effort", color = "#F0E68C")
plt.bar(x - 0.5 *bar_width, b_mae, bar_width, label="Best-Possible", color ="#66c2a5")
plt.bar(x + 0.5 * bar_width, c_mae, bar_width, label="Best-Effort", alpha=0.8, color= "#4169E1")
plt.xticks(x, APP)
# plt.bar(x + bar_width, w_df1["mae"], bar_width, label="Least-Effort", color = "#F0E68C")
plt.legend()
plt.xlabel("Applications")
plt.ylabel("MAE")
plt.grid(linestyle = '-.')
# plt.show()
os.chdir(r"plot")
plt.savefig("Best_Worst.pdf", bbox_inches='tight')
# %%

# plt.plot(x, b_df1["01_acc"], label="best")
# plt.plot(x, c_df1["01_acc"], label="cluster")
# plt.plot(x, w_df1["01_acc"], label="worst")
# plt.xticks(x, b_df1["app"])
# plt.legend()
# plt.show()

# %%
