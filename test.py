import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import GetLoader

batch_size = 65
DATA = pd.read_csv('data/minmax_data.csv', header=None, index_col=None).values
DATA = np.flipud(DATA)
data_D = DATA[:, 1:]
data_L = DATA[:, 1]
data_train, data_test, label_train, label_test = train_test_split(data_D, data_L, test_size=0.5)
train_set = GetLoader(data_train, label_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = GetLoader(data_test, label_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

for idx, (real, _), (fake, _) in enumerate(train_loader, test_loader):
    print('real_shape', real.shape)
    print('fake_shape', fake.shape)