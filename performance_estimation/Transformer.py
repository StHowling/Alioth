import pickle
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import pandas as pd
from datetime import date
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False

torch.manual_seed(0)
np.random.seed(0)
# %%
input_window = 100
output_window = 5
batch_size = 32 # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ("cpu")

# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# %%
class TransAm(nn.Module):
    def __init__(self, feature_size=10, num_layers=3, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, feature_size)
        # self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            # print('a',src.size())
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        # print('j',src.size(),self.src_mask.size())
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)

        output = self.decoder(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

#%%

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, label_data, tw):
    inout_seq = []
    L = len(input_data)
    # print(L)
    for i in range(tw+1 - tw):
        train_seq = np.append(input_data[i:i + tw, :][:-output_window, :].tolist(), np.zeros((output_window, 219)).tolist(), axis=0)
        # print(train_seq.shape)
        # train_label = input_data[i:i + tw, :]
        # train_label = input_data[i:i + tw, :,i:i+tw]
        # train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = label_data[i:i+tw]
        # print(train_seq.shape,train_label.shape)
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq.tolist(), train_label))
        print(inout_seq)
    return torch.FloatTensor(inout_seq)


def get_data():
    # amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    # data = data_.loc[
    #     (data_["date"] >= pd.Timestamp(date(2014, 1, 1))) & (data_["date"] <= pd.Timestamp(date(2014, 2, 10)))]
    # data = data.loc[:, "MT_200":  "MT_209"]

    # print(type(data_))
    # print(type(data_.tolist()))
    # series = data_.to_numpy()
    # print('a',series.shape)
    # amplitude = scaler.fit_transform(series)
    # print('b', data_.shape)
    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # print(amplitude.shape)
    sampels = int(len(data_)*0.8)
    train_data = data_[:sampels]
    train_label = label_[:sampels]
    # print('train', train_data.shape)
    test_data = data_[sampels:]
    test_label = label_[:sampels]
    # print('test', test_data.shape)
    # print(train_data.shape,test_data.shape)
    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    # print('c',train_data.shape)

    train_sequence = create_inout_sequences(train_data, train_label,input_window)
    # print('a',train_sequence.size())
    train_sequence = train_sequence[:-output_window]  # todo: fix hack?
    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, test_label, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return train_sequence.to(device), test_data.to(device), scaler


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1)).squeeze()  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1)).squeeze()
    return input, target

# %%
def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# %%
def plot_and_loss(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)

            # look like the model returns static values for the output window
            output = eval_model(data)

            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()), 0)  # todo: check this. -> looks good to me
            # test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)
            # truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    print(test_result.size(), truth.size())
    test_result = scaler.inverse_transform(test_result.reshape(-1, 1)).reshape(-1)
    truth = scaler.inverse_transform(truth.reshape(-1, 1)).reshape(-1)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth[:500], color="blue")
    pyplot.axhline(y=0, color='k')
    pyplot.xlabel("Periods")
    pyplot.ylabel("Y")
    pyplot.savefig('graph/transformer-epoch%d.png' % epoch)
    pyplot.close()
    return total_loss / i


def predict_future(eval_model, data_source, steps, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    data = data.cpu().view(-1)

    data = scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)
    pyplot.plot(data, color="red")
    pyplot.plot(data[:input_window], color="blue")
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    pyplot.savefig('graph/transformer-future%d.png' % epoch)
    pyplot.close()


# entweder ist hier ein fehler im loss oder in der train methode, aber die ergebnisse sind unterschiedlich
# auch zu denen der predict_future
def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            print(output[-output_window:].size(), targets[-output_window:].size())
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)

def plot(eval_model, data_source, epoch, scaler):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            data = data.unsqueeze(1)
            target = target.unsqueeze(1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1, :].squeeze(1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1, :].squeeze(1).cpu()), 0)

    # test_result = test_result.cpu().numpy()
    len(test_result)

    test_result_ = scaler.inverse_transform(test_result[:700])
    truth_ = scaler.inverse_transform(truth)
    print(test_result.shape, truth.shape)
    for m in range(9):
        test_result = test_result_[:, m]
        truth = truth_[:, m]
        fig = pyplot.figure(1, figsize=(20, 5))
        fig.patch.set_facecolor('xkcd:white')
        pyplot.plot([k + 510 for k in range(190)], test_result[510:], color="red")
        pyplot.title('Prediction uncertainty')
        pyplot.plot(truth[:700], color="black")
        pyplot.legend(["prediction", "true"], loc="upper left")
        ymin, ymax = pyplot.ylim()
        pyplot.vlines(510, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
        pyplot.ylim(ymin, ymax)
        pyplot.xlabel("Periods")
        pyplot.ylabel("Y")
        pyplot.show()
        pyplot.close()
    return total_loss / i

#%%
dir_ = '/home/yyx/interference_prediction/Alioth/Result/DAE_Pred11.npy'
dir2 = '/home/yyx/DP/classified1/1_data.csv'
data_ = np.load(dir_,allow_pickle=True)
label_ = pd.read_csv(dir2).drop(['workload_intensity'], axis=1)
label_ = torch.from_numpy(label_.values).float()
label_ = (label_[:,-1] + 1).numpy().tolist()
# print(label_)
train_data, val_data, scaler = get_data()
# print(train_data.size())
# print(train_data.size(), label_data.size())
tr,te = get_batch(train_data, 0,batch_size)
# print(tr.shape,te.shape)

model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

best_val_loss = float("inf")
epochs = 100  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % 10 is 0):
        val_loss = plot(model, val_data, epoch, scaler)
        # predict_future(model, val_data,200,epoch,scaler)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                time.time() - epoch_start_time),
                                                                                                  val_loss,
                                                                                                  math.exp(val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step()