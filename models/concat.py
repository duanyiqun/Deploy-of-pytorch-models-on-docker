import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import SDMv5
import torch.backends.cudnn as cudnn

densenet = models.densenet161(pretrained=True)
in_features = densenet.classifier.in_features
densenet.classifier = nn.Linear(in_features, 2, bias=True)

def densenet_Porn_Com():
    return densenet

def load_dense_fix(trainpath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #best_acc = 0  # best test accuracy
    #start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    net = densenet_Porn_Com()
    net = net.to(device)
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

"""
class Catnet(nn.Module):
    def __init__(self, num_classes=2, trainpath = './train/dense/dense.plk', trainpath2 = './train/sdmnv5/sdmnv.plk'):
        super(Catnet, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #best_acc = 0  # best test accuracy
    #start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        net = densenet_Porn_Com()
        net = net.to(device)
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
        in_features = net.module.classifier.in_features
        net.module.classifier = nn.Linear(in_features, 1000, bias=True)
        print(net)
        nets=load_sdm('./train/sdmnv5/sdmnv.plk')
        in_features = nets.module.linear.in_features
        nets.module.linear=nn.Linear(in_features,1000,bias=True)
        print(nets)
        for i,p in enumerate(net.module.parameters()):
            p.require_grad = False
        net.module.classifier.require_grad = True
        for i,p in enumerate(nets.module.parameters()):
            p.require_grad = False
        nets.module.linear.require_grad = True
        self.nets = nets
        self.netd = net

    def forward(self,x1,x2):
        #out = torch.cat(self.netd(x),self.nets(x)).contiguous()
        out1 = self.netd(x1)
        out2 = self.nets(x2)
        return out1 , out2

"""


class Catnet(nn.Module):
    def __init__(self, dense, sdmn):
        super(Catnet, self).__init__()
        print(dense)
        print(sdmn)
        self.densepart = nn.Sequential(*list(dense.children())[:-1])
        self.sdpart = nn.Sequential(*list(sdmn.children())[:-1])
    
    def forward(self, x1, x2):
        out1 = self.densepart(x1)
        out2 = self.sdpart(x2)
        return out1 , out2

class Net(nn.Module):
    def __init__(self , model):
        super(Net, self).__init__()
        #取掉model的后两层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        
        self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        self.pool_layer = nn.MaxPool2d(32)  
        self.Linear_layer = nn.Linear(2048, 8)
        
    def forward(self, x):
        x = self.resnet_layer(x)
 
        x = self.transion_layer(x)
 
        x = self.pool_layer(x)
 
        x = x.view(x.size(0), -1) 
 
        x = self.Linear_layer(x)
        
        return x


        
#cc=Catnet(netd , nets)


nums = []

def extract(m):
    global sparses
    global nums
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nums.append(torch.numel(m.weight.data))

def test_catnet():
    #netd = load_dense_fix('./train/dense/dense.plk')
    nets = load_sdm('./train/sdmnv5/sdmnv.plk')
    #net = Catnet(netd , nets)
    x1 = torch.randn(1,3,224,224)
    x2 = torch.randn(1,3,96,96)
    y1 = nets(Variable(x1))
    #y1,y2 = net(Variable(x1),Variable(x2))
    #print(y[0].size(),y[1].size(),y[2].size())
    nets.apply(extract)
    print(nums)
    print('sum of network param{}'.format(sum(nums)))
    print(y1.size())
    #print(y2.size())
    #print(nets)


test_catnet()
#netd=load_dense_fix('./train/dense/dense.plk')
#nets=nets=load_sdm('./train/sdmnv5/sdmnv.plk')