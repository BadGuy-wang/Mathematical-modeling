import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import deal_data, GetLoader, plot_curve

# %%

filename = 'data/minmax_data.csv'
# 定义超参数
device = torch.device('cuda')
epoch = 2000
batch_size = 65
lr = 0.0001


class LSTMpred(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(LSTMpred, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, seq):
        seq = seq.view(1, len(seq), 17)
        lstm_out, (hidden, cell) = self.lstm(seq.float())
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        outdat = self.fc(hidden)
        return outdat


# %%
model = LSTMpred(17, 9)
model.to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
DATA = np.flipud(DATA)
data_D = DATA[:, 1:]
data_L = DATA[:, 1]
train_set = GetLoader(data_D, data_L)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
# %%
train_loss = []
for e in tqdm(range(epoch)):
    temp_loss = []
    for idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()

        modout = model(X)
        modout = modout.view(X.shape[0])

        loss = loss_function(modout, Y.float())
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())
    train_loss.append(np.mean(temp_loss))

torch.save(model.state_dict(), 'LSTM.pth')
plot_curve(train_loss, model_name='LSTM', epoch=epoch, lr=lr)

# %%
# train_set = GetLoader(data_D, data_L)
# train_loader = DataLoader(train_set, batch_size=65, shuffle=False)
predDat = []
model.eval()
model.to(torch.device('cpu'))
for idx, (X, Y) in enumerate(train_loader):
    with torch.no_grad():
        X = X.to(torch.device('cpu'))
        out = model(X)
        predDat.append(out.numpy())

new_label = []
for i in range(len(predDat)):
    new_label.extend(predDat[i].tolist())

df = pd.DataFrame(new_label)
df.to_csv('data/LSTM_result_run_epoch'+str(epoch)+'.csv', header=None, index=None)

figsize = 11, 9
figure, ax = plt.subplots(figsize=figsize)
from matplotlib.pyplot import MultipleLocator

x_major_locator = MultipleLocator(25)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(0.25)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
plt.plot(np.arange(325), data_L, )
plt.plot(np.arange(325), new_label, '-.')
plt.title("LOSS_RUN_EPOCH" + str(epoch)+"lr="+str(lr))
l1, = plt.plot(np.arange(325), data_L, 'r')
l2, = plt.plot(np.arange(325), new_label, 'g-.')
plt.legend(handles=[l1, l2], labels=['real', 'predict'], loc='lower right')

figure.savefig("pic/LSTM_RUN_EPOCH" + str(epoch)+'.png', dpi=600,format='png')
# plt.show()