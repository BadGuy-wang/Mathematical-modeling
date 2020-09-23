import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from utils import GetLoader
from GAN_model import discriminator, generator

# 定义超参数


epoch = 100
batch_size = 15
# 创建数据加载器
real_DATA = pd.read_csv('data/problem_4.csv', header=None, index_col=None).values
real_D = real_DATA[:, 1:]
real_L = real_DATA[:, 1]
fake_DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
fake_D = fake_DATA[:315, 1:]
fake_L = fake_DATA[:315, 1]
fake_set = GetLoader(fake_D, fake_L)
fake_loader = DataLoader(fake_set, batch_size=batch_size, shuffle=True)
# real_set = GetLoader(real_D, real_L)
# fake_loader = DataLoader(real_set, batch_size=batch_size, shuffle=True)
# 实例化网络
D = discriminator()
G = generator()
# 使用GPU训练
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

# 定义损失函数与优化器
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# 开始训练
for e in range(epoch):
    for i, (data, _) in enumerate(fake_loader):
        num_data = data.size(0)
        # =================train discriminator
        real_data = Variable(torch.tensor(real_D)).cuda()
        real_label = Variable(torch.ones(num_data)).cuda()
        fake_label = Variable(torch.zeros(num_data)).cuda()

        # compute loss of real_img
        real_out = D(real_data)
        d_loss_real = criterion(real_out, real_label)
        real_scores = torch.mean(real_out) # closer to 1 means better
        # print(real_scores.dtype)

        # compute loss of fake_img
        # z = Variable(torch.randn(num_data, 1)).cuda()
        z = Variable(data).cuda()
        fake_data = G(z)
        fake_out = D(fake_data)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = torch.mean(fake_out) # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        # z = Variable(torch.randn(num_data, 1)).cuda()
        z = Variable(data).cuda()
        fake_data = G(z)
        output = D(fake_data)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 5 == 0:
            print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                  'D real: {:.6f}, D fake: {:.6f}'.format(
                e, epoch, d_loss.data, g_loss.data,
                real_scores, fake_scores))
#%%
G.to(torch.device('cpu'))
origin_data = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
# origin_D = origin_data[:, 1:]
origin_D = origin_data[133, 1:]
df_o = pd.DataFrame(origin_D)
df_o.to_csv('133_data.csv', header=None, index=None)
# reslut_list = []
# for i in range(origin_D.shape[0]):
#     z = Variable(torch.tensor(origin_D[i]))
#     fake_data = G(z)
#     temp_list = list(fake_data.detach().numpy())
#     reslut_list.append(temp_list)
# reslut_list = np.array(reslut_list).reshape((325, 17))
z = Variable(torch.tensor(origin_D))
fake_data = G(z)
df = pd.DataFrame(fake_data.detach().numpy())
# df = pd.DataFrame(reslut_list)
df.to_csv('generator_data/G_run_133.csv', header=None, index=None)