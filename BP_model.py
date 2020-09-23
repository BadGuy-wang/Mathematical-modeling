import torch.nn as nn
import torch



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