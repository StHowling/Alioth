# %%
# processing total 28000+ raw events
import pandas as pd
import numpy as np
import os

PATH = "../"
# %%
df = pd.read_csv(os.path.join(PATH, "all_events.txt"), sep='\t')
print(df.shape)

# %%
df.drop_duplicates(subset=['ActualCode'], inplace=True)
print(df.shape)
# %%
df = df[df['Desc'] != "tracepoint"]
print(df.shape)
# %%
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
df.to_csv(os.path.join(PATH, "all_events_drop_duplicate.txt"), sep='\t', index=False)
# %%
df["ActualCode"].to_csv(os.path.join(PATH, "total_code"), index=False, header=False, line_terminator='\n')
# %%
# ----------------------------------------
df = pd.read_csv(os.path.join(PATH, "all_events_drop_duplicate.txt"), sep='\t')
# %%
pmuname = ['perf']
selected_df = df[df['PMUName'].isin(pmuname)]
sdf = selected_df[['ActualCode','Name[Umask]']]
# %%
sdf["ActualCode"].to_csv(os.path.join(PATH, "total_code"), index=False, header=False, line_terminator='\n')
sdf.to_csv(os.path.join(PATH, "select_code"), index=False, header=False, line_terminator='\n')
# %%
# -------------------------------------------------
total_code_selected = {}
with open(os.path.join(PATH, "total_code_selected"), "r") as f:
    for line in f.readlines():
        line = line.strip()
        total_code_selected[line] = df[df["ActualCode"] == line]["Name[Umask]"].iloc[0]

# %%
total_code_selected_df = pd.DataFrame.from_dict(total_code_selected, orient="index")
total_code_selected_df.to_csv(os.path.join(PATH, "total_code_selected.txt"), header=False, line_terminator='\n')
# %%
# df[df["ActualCode"] == "r52003c"]["Name[Umask]"].iloc[0]