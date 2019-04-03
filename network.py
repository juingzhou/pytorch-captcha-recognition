import torch.nn as nn
import torch.nn.functional as F
import captcha_generate


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )



        self.fc11 = nn.Sequential(
            
            nn.Linear(1536, captcha_generate.ALL_CHAR_SET_LEN),

        )
        self.fc12 = nn.Sequential(
            
            nn.Linear(1536, captcha_generate.ALL_CHAR_SET_LEN),

        )
        self.fc13 = nn.Sequential(
            
            nn.Linear(1536, captcha_generate.ALL_CHAR_SET_LEN),

        )
        self.fc14 = nn.Sequential(
            
            nn.Linear(1536, captcha_generate.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = out.view(-1, 1536)
        out = F.dropout(out, p=0.25, training=self.training)
        out = F.relu(out)
        out0 = F.softmax(self.fc11(out),dim=1)
        out2 = F.softmax(self.fc12(out),dim=1)
        out3 = F.softmax(self.fc13(out),dim=1)
        out4 = F.softmax(self.fc14(out),dim=1)
        return out0, out2, out3, out4


