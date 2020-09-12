import torch.nn as nn
import numpy as np

#1x3x48x120
class ONet(nn.Module):
    def __init__(self,keypoint_num):
        super(ONet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 32, 3, 1)),
                ('prelu1', nn.PReLU(32)),
                ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv2', nn.Conv2d(32, 64, 3, 1)),
                ('prelu2', nn.PReLU(64)),
                ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
                ('conv3', nn.Conv2d(64, 64, 3, 1)),
                ('prelu3', nn.PReLU(64)),
                ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
                ('conv4', nn.Conv2d(64, 128, 2, 1)),
                ('prelu4', nn.PReLU(128)),
                ('flatten', Flatten()),
                ('conv5', nn.Linear(1152, 256)),
                ('drop5', nn.Dropout(0.25)),
                ('prelu5', nn.PReLU(256)),
            ]))
        self.landmarks = nn.Linear(256, keypoint_num*2)

    def forward(self, x):
        x = self.features(x)
        lm=self.landmarks(x)
        return lm

if __name__ == '__main__':
    net =ONet(4)
