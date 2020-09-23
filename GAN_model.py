import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(17, 8, bias=True),
            nn.Linear(8, 4, bias=True),
            nn.Linear(4, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x.float())
        return x.view(x.size(0))


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(17, 17),
            # nn.ReLU(True),
            # nn.Linear(9, 18),
            nn.Tanh()
        )

    def forward(self, x):
        # x = self.gen(x.float())
        return self.gen(x.float())
