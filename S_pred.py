import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from BP_model import BPNet
from utils import GetLoader, plot_curve
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 65
EPOCH = 2000
lr = 0.005
DEVICE = torch.device('cuda')

DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
data_D = DATA[:, 1:]
DATA = pd.read_csv('S.csv', header=None, index_col=None).values
# DATA = np.flipud(DATA)
min_max_scaler = MinMaxScaler().fit(DATA)
DATA = min_max_scaler.transform(DATA)
data_L = DATA
train_set = GetLoader(data_D, data_L)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

Snet = BPNet()
optimizer = optim.Adam(Snet.parameters(), lr=lr)
loss_function = nn.MSELoss()
Snet.to(DEVICE)
train_loss = []
for e in tqdm(range(EPOCH)):
    Snet.train()
    temp_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        out = Snet(data)
        loss = loss_function(target.float(), out)
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())
    train_loss.append(np.mean(temp_loss))

plot_curve(train_loss, model_name='S_loss', epoch=EPOCH, lr=lr)


# DATA = pd.read_csv('data/optim_data.csv', header=None, index_col=None).values
# min_max_scaler = MinMaxScaler().fit(DATA)
# DATA = min_max_scaler.transform(DATA)
# train_set = GetLoader(DATA, data_L)
# train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

device = torch.device('cpu')
Snet.to(device)
Snet.eval()
pred_labels = []
for batch_idx, (data, _) in enumerate(train_loader):
    with torch.no_grad():
        data = data.to(device)
        out = Snet(data)
        pred_labels.append(out.numpy())

new_label = []
for i in range(len(pred_labels)):
    new_label.extend(pred_labels[i].tolist())

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
l1, = plt.plot(np.arange(325), data_L, 'r')
l2, = plt.plot(np.arange(325), new_label, 'g-.')
plt.legend(handles=[l1, l2], labels=['S real', 'S predict'], loc='lower right')

plt.title("S" + str(EPOCH)+"lr="+str(lr))
figure.savefig("pic/S_pre" + str(EPOCH)+'.png', dpi=600,format='png')