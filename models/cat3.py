import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from models import SDMv5_f
#import SDMv5_f
import torch.backends.cudnn as cudnn
#from models import densenet
import random

def load_sdm(trainpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #best_acc = 0  # best test accuracy
    #start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net = SDMv5_f.modulenet_Porn_Com()
    net = net.to(device)
    #net = torch.nn.DataParallel(net)
    """
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    else:
        net = torch.nn.DataParallel(net)
    """

    assert os.path.isfile(trainpath), 'Error: no trained model directory found!'
    if device == 'cpu':
        checkpoint = torch.load(trainpath,map_location='cpu')
        #checkpoint = torch.load(trainpath) 
    else:
        checkpoint = torch.load(trainpath)  
    
    model_dict = net.state_dict()
    pretrained_dict =  {k: v for k, v in checkpoint['net'].module.items() if k in model_dict}
    #net.load_state_dict(checkpoint['net'])
    model_dict.update(pretrained_dict)
    best_acc = checkpoint['acc']
    net.load_state_dict(model_dict)   
    net.load_state_dict(checkpoint['net'])
    
    return net

class Catnet(nn.Module):
    def __init__(self):
        super(Catnet, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nets = SDMv5_f.modulenet_Porn_Com()
        nets.to(device)
        #31212
        netd =  models.densenet161(pretrained=True)
        netd.to(device)
        in_features = netd.classifier.in_features
        netd.classifier = nn.Linear(in_features, 2000, bias=True)

        self.nets = nets
        self.netd = netd
        self.linear_1 = nn.Linear(4000,2,bias=True)
        #self.linear_2 = nn.Linear(2000,2,bias=True)

    
    def forward(self, x1):
        w = random.randint(1, 127)
        #print(w)
        a = []
        for i in range(96):
            a.append(w+i)

        indices = torch.cuda.LongTensor(a)
        #indices = torch.LongTensor(a)
        x2 = torch.index_select(x1, 2 ,indices)
        x2 = torch.index_select(x2, 3, indices)
        #print(x2.size())

        out1 = self.nets(x2)
        out2 = self.netd(x1)
        #print(out1.size())
        #print(out2.size())
        out = torch.cat((out1,out2),dim=1)
        out = self.linear_1(out.contiguous())
        #out = self.linear2(out.contiguous())
        
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