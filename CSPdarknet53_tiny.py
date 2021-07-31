import torch
import torch.nn as nn
import math

class Basicconv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1):
        super(Basicconv, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.leakyRelu=nn.LeakyReLU(0.1)
    def forward(self,x):
        return self.leakyRelu(self.bn(self.conv(x)))

class resblock_body(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(resblock_body, self).__init__()
        self.out_channels=out_channels
        self.conv1=Basicconv(in_channels,out_channels,3)
        self.conv2=Basicconv(out_channels//2,out_channels//2,3)
        self.conv3=Basicconv(out_channels//2,out_channels//2,3)
        self.conv4=Basicconv(out_channels,out_channels,1)
        self.maxPool=nn.MaxPool2d(2)

    def forward(self,x):
        x=self.conv1(x)

        route=x

        _,x=torch.split(x,self.out_channels//2,dim=1)
        x=self.conv2(x)

        route1=x

        x=self.conv3(x)
        x=torch.cat((x,route1),dim=1)
        x=self.conv4(x)

        feat=x

        x=torch.cat((x,route),dim=1)
        x=self.maxPool(x)
        return x,feat#NCHW   ->N,2C,H/2,W/2 |||   ->N,C,H,W

class CSPdarknet(nn.Module):
    def __init__(self):
        super(CSPdarknet, self).__init__()
        self.conv1=Basicconv(3,32,3,2)
        self.conv2=Basicconv(32,64,3,2)
        self.resblock1=resblock_body(64,64)
        self.resblock2=resblock_body(128,128)
        self.resblock3=resblock_body(256,256)
        self.conv3=Basicconv(512,512,3)

        #权重初始化，使其符合正态分布规律
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))#https://blog.csdn.net/tsq292978891/article/details/79382306
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x,_=self.resblock1(x)
        x,_=self.resblock2(x)

        x,feat1=self.resblock3(x)

        x=self.conv3(x)
        feat2=x
        return feat1,feat2 #[N,256,26,26],[N,512,13,13]
def darknet53_tiny(pretained):
    model=CSPdarknet()
    if pretained:
        if isinstance(pretained,str):
            model.load_state_dict(torch.load(pretained))
        else:
            raise Exception(f"darknet request a pretrained path. got[{pretained}]")
    return model


if __name__ == '__main__':
    a=torch.arange(64,dtype=torch.float).reshape(1,4,4,4)
    re=resblock_body(4,4)
    print(re(a)[0].shape,re(a)[1].shape)
    b=torch.ones([1,3,416,416])*3
    asd=CSPdarknet()
    print(asd(b)[0].shape)
    print(asd(b)[1].shape)