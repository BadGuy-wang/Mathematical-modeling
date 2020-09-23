import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from BP_model import BPNet
from utils import GetLoader
import numpy as np
import pandas as pd
from torch.autograd import Variable
# #%%
# DATA = pd.read_csv('generator_data/G_run_origin.csv', header=None, index_col=None).values
# # DATA = DATA[np.lexsort(DATA.T)] #以最后一列为标准顺序排序（从小到大）
# data_D = DATA[:, 1:]
# data_L = DATA[:, 1]
# #%%
# train_set = GetLoader(data_D, data_L)
# train_loader = DataLoader(train_set, batch_size=65, shuffle=False)
# device = torch.device('cpu')
# net = BPNet()
# net.load_state_dict(torch.load('Baseline.pth'))
# net.to(device)
#
# pred_labels = []
# for batch_idx, (data, _) in enumerate(train_loader):
#     with torch.no_grad():
#         data = data.to(device)
#         out = net(data)
#         pred_labels.append(out.numpy())
#
# new_label = []
# for i in range(len(pred_labels)):
#     new_label.extend(pred_labels[i].tolist())
# # 标题显示中文
# #%%
# plt.plot(np.arange(325), data_L)
# plt.plot(np.arange(325), new_label, "-.")
# plt.show()


net = BPNet()
net.to(torch.device('cpu'))
net.load_state_dict(torch.load('checkpionts/Baseline.pth'))
net.eval()

# DATA = pd.read_csv('generator_data/G_run_origin.csv', header=None, index_col=None).values
# data_D = DATA
# DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
# DATA = np.flipud(DATA)
# data_L = DATA[:, 1]
# train_set = GetLoader(data_D, data_L)
# train_loader = DataLoader(train_set, batch_size=65, shuffle=False)
DATA = pd.read_csv('data/optim_data.csv', header=None, index_col=None).values
min_max_scaler = MinMaxScaler().fit(DATA)
DATA = min_max_scaler.transform(DATA)
DATA = np.flipud(DATA)
data_L = DATA[:, 1]

train_set = GetLoader(DATA, data_L)
train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

pred_labels = []
for idx, (data, _) in enumerate(train_loader):
    with torch.no_grad():
        out = net(data)
        pred_labels.append(out.detach().numpy())

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
# l1, = plt.plot(np.arange(325), data_L, 'r')
l2, = plt.plot(np.arange(51), new_label, 'g-.')
plt.legend(handles=[l2], labels=['RON predict'], loc='lower right')


figure.savefig("pic/RON.png", dpi=600,format='png')

