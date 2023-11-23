# %%
import os
import time
import argparse
import logging
import pickle
import torch
# set torch random seed so that the result is reproducible
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, random_split
from utils import *
from model import DAE_random_dataset, DAE_new
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# %%
# setting up logger
logfile = "./log/Alioth DAE.log"
make_file_dir(logfile)
logger = logging.getLogger("Alioth DAE logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(logfile)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# %%
def get_parameters():
    parser = argparse.ArgumentParser(description="Alioth offline version")
    parser.add_argument('--save_finalresult', type=int, default=1, metavar='N',
                        help='save final result or not')
    # args = parser.parse_args([])
    args = parser.parse_args([])
    return args


args = get_parameters()
logger.info(args)
batch_size = 32
epoches = 150
lr = 0.001
train_ratio = 0.8
data_dir = "../data/output"
model_save_path = "model/Alioth"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
dataset = DAE_random_dataset(os.path.join(data_dir, "DAE_features_all_intall_warming3_nosliding_all.csv"))
train_len = int(len(dataset) * train_ratio)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
model = DAE_new(len(dataset[0][0])).to(device)
logger.info(f"Model dimension: {len(dataset[0][0])}")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(epoch):
    model.train()
    total_count = 0
    total_loss = 0.
    for idx, (data, nostress_data) in enumerate(train_dataloader):
        data = data.to(device)
        nostress_data = nostress_data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        loss = criterion(output, nostress_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.shape[0]
        total_count += data.shape[0]
    total_loss /= total_count
    return total_loss

def test(epoch):
    model.eval()
    total_count = 0
    total_loss = 0.
    for data, nostress_data in test_dataloader:
        data = data.to(device)
        nostress_data = nostress_data.to(device)
        outputs = model(data)
        loss = criterion(outputs, nostress_data)
        total_loss += loss.item() * data.shape[0]
        total_count += data.shape[0]
    total_loss /= total_count
    return total_loss

def draw_loss_figure(train_losses, test_losses):
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.plot(train_losses, c="#EA906C")
    ax1.set_xlabel("train losses")
    ax2.plot(test_losses, c="#7ED7C1")
    ax2.set_xlabel("test losses")
    plt.savefig(os.path.join(model_save_path, "train_result.png"), bbox_inches="tight")

# %%
train_losses = []
test_losses = []
for epoch in tqdm(range(epoches), desc="Epoch", leave=False):
    train_loss = train(epoch)
    test_loss = test(epoch)
    logger.info("Epoch {}: train loss {}, test loss {}".format(epoch, train_loss, test_loss))
    train_losses.append(train_loss)
    test_losses.append(test_loss)

draw_loss_figure(train_losses, test_losses)

if args.save_finalresult:
    torch.save(model.state_dict(), os.path.join(model_save_path, "DAE.pt"))

# # %%
# for idx, (data, nostress_data) in enumerate(train_dataloader):
#     print(1)
#     break
# data = data.to(device)
# nostress_data = nostress_data.to(device)
# optimizer.zero_grad()
# output = model(data)

# # %%
# model_save_path = "model/Alioth"
# e = Encoder(1090, 128)
# dae_paras = torch.load(os.path.join(model_save_path, "DAE.pt"))
# dae_paras = {k.replace("encoder.", ""): v for k, v in dae_paras.items() if k.replace("encoder.", "") in e.state_dict()}
# e.load_state_dict(dae_paras)
# # for i in dae_paras.keys():
# #     for j in e.state_dict().keys():
# #         if "encoder" in i and j in i:
# #             e.load_state_dict(dae_paras[i])