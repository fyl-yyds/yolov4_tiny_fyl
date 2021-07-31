import torch
import torch.nn as nn
from CSPdarknet53_tiny import Basicconv,darknet53_tiny
from attention_fyl import se_block,cbam_block,eca_block

class upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample, self).__init__()
        self.upsample=nn.Sequential(
            Basicconv(in_channels,out_channels,1),
            nn.Upsample(scale_factor=2,mode="nearest")
        )
    def forward(self,x):
        return self.upsample(x)

class yolohead(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels):
        super(yolohead, self).__init__()
        self.head=nn.Sequential(
            Basicconv(in_channels,mid_channels,3),
            nn.Conv2d(mid_channels,out_channels,1)
        )
    def forward(self,x):
        return self.head(x)

class Yolobody(nn.Module):
    def __init__(self,num_anchors,num_class,phi):
        super(Yolobody, self).__init__()
        Attention=[se_block,cbam_block,eca_block]
        if phi>=4:
            assert AssertionError("phi必须为(0、1、2、3)其中的某个值")
        self.phi=phi
        self.net=darknet53_tiny(None)
        self.conv1=Basicconv(512,256,1)
        self.yolohead13=yolohead(256,512,num_anchors*(num_class+5))
        self.upsample=upsample(256,128)
        self.yolohead26=yolohead(384,256,num_anchors*(num_class+5))

        if 1<=self.phi and self.phi<=3:
            self.attention512=Attention[self.phi-1](512)
            self.attention256=Attention[self.phi-1](256)
            self.attention128=Attention[self.phi-1](128)


    def forward(self,x):
        # feat1:[N,256,26,26],feat2:[N,512,13,13]
        feat1,feat2=self.net(x)
        if 1<=self.phi and self.phi<=3:
            feat1=self.attention256(feat1)
            feat2=self.attention512(feat2)

        P13=self.conv1(feat2)
        out13=self.yolohead13(P13)

        up_26=self.upsample(P13)

        if 1 <= self.phi and self.phi <= 3:
            up_26=self.attention128(up_26)
        P26=torch.cat((feat1,up_26),dim=1)
        out26=self.yolohead26(P26)
        return out13,out26 #[N,30,13,13],[N,30,26,26]

if __name__ == '__main__':
    a=torch.ones(1,3,416,416)
    you=Yolobody(3,5,2)
    q,w=you(a)
    print(q.shape)
    print(w.shape)