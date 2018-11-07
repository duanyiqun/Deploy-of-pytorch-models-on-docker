import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

densenet = models.densenet161(pretrained=True)
in_features = densenet.classifier.in_features
densenet.classifier = nn.Linear(in_features, 2, bias=True)

def densenet_Porn_Com():
    return densenet

nums = []

def extract(m):
    global sparses
    global nums
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nums.append(torch.numel(m.weight.data))

def test_densenet():
    net = densenet_Porn_Com()
    x = torch.randn(1,3,224,224)
    y= net(Variable(x))
    #print(y[0].size(),y[1].size(),y[2].size())
    net.apply(extract)
    print(nums)
    print('sum of network param{}'.format(sum(nums)))
    print(y.size())
    print(net)

#test_densenet()