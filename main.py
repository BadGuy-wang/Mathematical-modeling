import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from utils import GetLoader, plot_curve
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


# %%
class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(17, 9, bias=True)
        self.fc2 = nn.Linear(9, 4, bias=True)
        self.fc3 = nn.Linear(4, 1, bias=True)
        # self.fc4 = nn.Linear(3, 1, bias=True)

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        x = self.fc3(x)
        x = x.view(x.shape[0])
        return x


# %%
BATCH_SIZE = 65
EPOCH = 1900
lr = 0.0001
DEVICE = torch.device('cuda')
DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
DATA = np.flipud(DATA)
data_D = DATA[:, 1:]
data_L = DATA[:, 1]
data_train, data_test, label_train, label_test = train_test_split(data_D, data_L, test_size=0.5)
train_set = GetLoader(data_train, label_train)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# %%
net = BPNet()
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_function = nn.MSELoss()
net.to(DEVICE)
train_loss = []
for e in tqdm(range(EPOCH)):
    net.train()
    temp_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        out = net(data)
        loss = loss_function(target.float(), out)
        loss.backward()
        optimizer.step()
        temp_loss.append(loss.item())
    train_loss.append(np.mean(temp_loss))
torch.save(net.state_dict(), 'checkpionts/Baseline.pth')
plot_curve(train_loss, model_name='BPNet', epoch=EPOCH, lr=lr)
# %%
train_set = GetLoader(data_D, data_L)
train_loader = DataLoader(train_set, batch_size=65, shuffle=False)

# DATA = pd.read_csv('generator_data/G_run_origin.csv', header=None, index_col=None).values
# # DATA = DATA[np.lexsort(DATA.T)] #以最后一列为标准顺序排序（从小到大）
# data_D = DATA
# train_set = GetLoader(data_D, data_L)
# train_loader = DataLoader(train_set, batch_size=65, shuffle=False)

device = torch.device('cpu')
net.to(device)
net.eval()
pred_labels = []
for batch_idx, (data, _) in enumerate(train_loader):
    with torch.no_grad():
        data = data.to(device)
        out = net(data)
        pred_labels.append(out.numpy())

new_label = []
for i in range(len(pred_labels)):
    new_label.extend(pred_labels[i].tolist())
df = pd.DataFrame(new_label)
df.to_csv('data/BP_result_run_epoch'+str(EPOCH)+'.csv', header=None, index=None)
# 标题显示中文
# %%
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
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
plt.legend(handles=[l1, l2], labels=['real', 'predict'], loc='lower right')

plt.title("BPNet_RUN_EPOCH" + str(EPOCH)+"lr="+str(lr))
figure.savefig("pic/BPNet_RUN_EPOCH" + str(EPOCH)+'.png', dpi=600,format='png')

# plt.show()
