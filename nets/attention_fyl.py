import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self,channels,ratio=16):
        super(se_block, self).__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio,bias=False), #Linear()要求输入格式NV
            nn.ReLU(inplace=True),
            nn.Linear(channels//ratio,channels,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        n,c,_,_=x.size()
        y=self.avgpool(x).reshape(n,c)
        y=self.fc(y).reshape(n,c,1,1)
        return y*x

class chaeenlsattention(nn.Module):
    def __init__(self,in_channels,ratio=8):
        super(chaeenlsattention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out+max_out)

class spatialattention(nn.Module):#(空间注意力)
    def __init__(self,kernel_size):
        super(spatialattention, self).__init__()
        assert kernel_size in (7,3),"卷积核必须是7或者3"
        padding = 3 if kernel_size ==7 else 1
        self.conv1=nn.Conv2d(2,1,kernel_size,(1,1),padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg=torch.mean(x,dim=1,keepdim=True)   #在通道维度上取一个均值
        max,_=torch.max(x,dim=1,keepdim=True)  #在通道维度上取一个最大值
        x=torch.cat((avg,max),dim=1)
        x=self.conv1(x)
        x=self.sigmoid(x)
        return x

class cbam_block(nn.Module):
    def __init__(self,in_channels,ratio=8,kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention=chaeenlsattention(in_channels,ratio=ratio)
        self.spatialattention=spatialattention(kernel_size=kernel_size)
    def forward(self,x):
        x=x*self.channelattention(x)
        x=x*self.spatialattention(x)
        return x

class eca_block(nn.Module):
    def __init__(self,in_channels,b=1,gamma=2):
        super(eca_block, self).__init__()
        kernel_size=int(abs((math.log(in_channels,2)+b)/gamma))#math.log(in_channels,2):2的多少次方=in_channels,int向下取整
        #in_channels=128的时候,kernel_size=4,256->4,512->5
        kernel_size=kernel_size if kernel_size%2 else kernel_size+1
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size,1,padding=(kernel_size-1)//2,bias=False)#这里是1维卷积！！！！
        #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        y=self.avgpool(x)
        y=self.conv(y.squeeze(-1).transpose(-1,-2))
        y=y.transpose(-1,-2).unsqueeze(-1)
        y=self.sigmoid(y)
        y=y.expand_as(x)
        return x*y

if __name__ == '__main__':
    a=torch.ones(2,256,20,20)
    qq=se_block(256)
    print(qq(a).shape)
    ww=chaeenlsattention(256)
    d=ww(a)
    print((d*a).shape)
    b=torch.mean((d*a),dim=1,keepdim=True)
    print(b.shape)
    # c,_=torch.max(d,dim=1,keepdim=True)
    # print(c.shape)
    ee=cbam_block(256)
    print(ee(a).shape)
    rr=eca_block(256)
    print(rr(a).shape)