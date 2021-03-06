import torch
import torch.nn as nn
from nets.CSPdarknet53_tiny import Basicconv,darknet53_tiny
from nets.attention_fyl import se_block,cbam_block,eca_block

class upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(upsample, self).__init__()
        self.upsample=nn.Sequential(
            Basicconv(in_channels,out_channels,1),
            nn.Upsample(scale_factor=2,mode="nearest")
        )
    def forward(self,x):
        return self.upsample(x)

# class yolo_head(nn.Module):
#     def __init__(self,in_channels,mid_channels,out_channels):
#         super(yolo_head, self).__init__()
#         self.head=nn.Sequential(
#             Basicconv(in_channels,mid_channels,3),
#             nn.Conv2d(mid_channels,out_channels,1)
#         )
#     def forward(self,x):
#         return self.head(x)
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        Basicconv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m

class Yolobody(nn.Module):
    def __init__(self,num_anchors,num_classes,phi):
        super(Yolobody, self).__init__()
        Attention=[se_block,cbam_block,eca_block]
        if phi>=4:
            assert AssertionError("phi必须为(0、1、2、3)其中的某个值")
        self.phi=phi
        self.backbone=darknet53_tiny(None)
        self.conv_for_P5=Basicconv(512,256,1)
        self.yolo_headP5=yolo_head([512, num_anchors * (5 + num_classes)],256)
        self.upsample=upsample(256,128)
        self.yolo_headP4=yolo_head([256, num_anchors * (5 + num_classes)],384)

        if 1<=self.phi and self.phi<=3:
            self.feat2_att=Attention[self.phi-1](512)
            self.feat1_att=Attention[self.phi-1](256)
            self.upsample_att=Attention[self.phi-1](128)


    def forward(self,x):
        # feat1:[N,256,26,26],feat2:[N,512,13,13]
        feat1,feat2=self.backbone(x)
        if 1<=self.phi and self.phi<=3:
            feat1=self.feat1_att(feat1)
            feat2=self.feat2_att(feat2)

        P13=self.conv_for_P5(feat2)
        out13=self.yolo_headP5(P13)

        up_26=self.upsample(P13)

        if 1 <= self.phi and self.phi <= 3:
            up_26=self.upsample_att(up_26)
        P26=torch.cat((feat1,up_26),dim=1)
        out26=self.yolo_headP4(P26)
        return out13,out26 #[N,30,13,13],[N,30,26,26]

if __name__ == '__main__':
    a=torch.ones(50,3,224,224)
    you=Yolobody(3,4,2)
    q,w=you(a)
    print(q.shape)
    print(w.shape)