from torch import nn
import torch.nn.functional as F


# Dueling Network
class Net(nn.Module):
    def __init__(self, n_in, n_mid_1, n_mid_2, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid_1)
        self.fc2 = nn.Linear(n_mid_1, n_mid_2)
        self.fc3_adv = nn.Linear(n_mid_2, n_out)
        self.fc3_v = nn.Linear(n_mid_2, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        # Not Relu
        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output


class CNNNet(nn.Module):
    def __init__(self, input_len, output_num, conv_size=(32, 64), fc_size=(512, 128), out_softmax=False):
        super(CNNNet, self).__init__()
        self.input_len = input_len
        self.output_num = output_num
        self.out_softmax = out_softmax

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_size[0], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_size[0], conv_size[1], kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(conv_size[1] * self.input_len * self.input_len, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.head = nn.Linear(fc_size[1], self.output_num)

    def forward(self, x):
        x = x.reshape(-1, 1, self.input_len, self.input_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        output = self.head(x)
        if self.out_softmax:
            output = F.softmax(output, dim=1)
        return output
