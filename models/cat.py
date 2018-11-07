import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import SDMv5
import torch.backends.cudnn as cudnn
import densenet
import random

def load_sdm(trainpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #best_acc = 0  # best test accuracy
    #start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net = SDMv5.modulenet_Porn_Com()
    net = net.to(device)
    #net = torch.nn.DataParallel(net)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    else:
        net = torch.nn.DataParallel(net)
    assert os.path.isfile(trainpath), 'Error: no trained model directory found!'
    if device == 'cpu':
        checkpoint = torch.load(trainpath,map_location='cpu')
        #checkpoint = torch.load(trainpath) 
    else:
        checkpoint = torch.load(trainpath)     
    net.load_state_dict(checkpoint['net'])
    
    return net

class Catnet(nn.Module):
    def __init__(self):
        super(Catnet, self).__init__()
        nets = load_sdm('./train/sdmnv5/sdmnv.plk')
        for i,p in enumerate(nets.module.parameters()):
            p.require_grad = False
        nets.module.linear.require_grad = True
        #31212
        netd =  models.densenet161(pretrained=True)
        in_features = netd.classifier.in_features
        netd.classifier = nn.Linear(in_features, 2208, bias=True)
        for i,p in enumerate(netd.parameters()):
            p.require_grad = False
        netd.classifier.require_grad = True

        self.nets = nets
        self.netd = netd
        self.linear1 = nn.Linear(2210,3000,bias=True)
        self.linear2 = nn.Linear(3000,1000,bias=True)
        self.linear3 = nn.Linear(1000,200,bias=True)
        self.linear4 = nn.Linear(200,2,bias=True)

    
    def forward(self, x1):
        w = random.randint(1, 127)
        #print(w)
        a = []
        for i in range(96):
            a.append(w+i)

        indices = torch.LongTensor(a)
        x2 = torch.index_select(x1, 2 ,indices)
        x2 = torch.index_select(x2, 3, indices)
        #print(x2.size())

        out1 = self.nets(x2)
        out2 = self.netd(x1)
        #print(out1.size())
        #print(out2.size())
        out = torch.cat((out1,out2),dim=1)
        out = self.linear1(out.contiguous())
        out = self.linear2(out.contiguous())
        out = self.linear3(out.contiguous())
        out = self.linear4(out.contiguous())
        
        return out
    

def pornvscom():
    return Catnet()

def test_catnet():
    cc = Catnet()
    print(cc)
    #x1 = torch.randn(1,3,96,96)
    x1 = torch.randn(1,3,224,224)
    y = cc(Variable(x1))
    print(y.size())
    print(y)

if __name__ == '__main__':
    test_catnet()