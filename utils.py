#%%
import pandas as pd
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def load_data(filename):
    FILENAME = filename
    data = pd.read_excel(FILENAME, header=0, usecols="A,D,H,I,J")
    DataArray = data.values
    Y = DataArray[:, 0]
    X = DataArray[:, 1:5]
    scaler_X = StandardScaler().fit(X)
    X = scaler_X.transform(X)

    return X, Y


def get_value_data(data, label):
    new_data_list = []
    for i in range(data.shape[0]):
        temp = list(data[i])
        temp.append(label[i])
        new_data_list.append(temp)
    df = pd.DataFrame(new_data_list)
    df.to_csv('data.csv', header=None, index=None)


class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


def plot_curve(data, model_name, epoch, lr):
    """
    Show training image

    Args:
        data:Two-dimensional array of image
    """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.title(model_name + "_LOSS_EPOCH" + str(epoch) + "lr=" + str(lr))
    fig.savefig('pic/'+model_name+str(epoch)+'.png', dpi=600, format='png')
    #
    # plt.show()


def deal_data(file_name):
    DATA = pd.read_csv(file_name, header=None, index_col=None).values
    DATA = np.flipud(DATA) # 数据逆序
    data_D = DATA[:, 3:]
    data_L = DATA[:, 2]
    pca = PCA(n_components=30)  # 使用PCA进行降维处理
    newX = pca.fit_transform(data_D)
    newX = pca.transform(data_D)
    return newX, data_L