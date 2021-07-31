import numpy as np
import torch
from nets.yolov4_tiny import Yolobody

def get_classes(classes_path):
    classes=[]
    with open(classes_path) as f:
        lines=f.readlines()
        for i in lines:
            classes.append(i.strip())
    return classes
def get_anchors(anchors_path):
    anchors=[]
    with open(anchors_path) as f:
        line=f.readline()
        line=list(line.strip().split(","))
        for i in line:
            a=float(i)
            anchors.append(a)
    return np.array(anchors).reshape((-1,3,2))

def weights_init(net,init_type="normal",init_gain=0.02):
    #选择初始化方法
    def init_func(m):
        classname=m.__class__.__name__#获取类名
        if hasattr(m,"weight") and classname.find("Conv")!=-1:
            #hasattr() 函数用于判断对象是否包含对应的属性。
            if init_type=="normal":#正态分布从初始化
                torch.nn.init.normal_(m.weight.data,0.0,init_gain)
            elif init_type=="xavier":#xavier初始化
                torch.nn.init.xavier_normal_(m.weight.data,gain=init_gain)
            elif init_type=="kaiming":#凯明初始化
                torch.nn.init.kaiming_normal_(m.weight.data,a=0,mode="fan_in")
            elif init_type=="orthogonal":#正交初始化
                torch.nn.init.orthogonal_(m.weight.data,gain=init_gain)
            else:
                raise NotImplemented(f"没有{init_type}这种初始化方法！")
        elif classname.find("BatchNorm2d")!=-1:
            torch.nn.init.normal_(m.weight.data,1.0,0.02)
            torch.nn.init.constant_(m.bias.data,0.0)#用0.0填充m.bias.data这个张量
    print(f"初始化网络用的是{init_type}方法")
    net.apply(init_func)


if __name__ == '__main__':
    #-------------------------------#
    #   所使用的注意力机制的类型
    #   phi = 0为不使用注意力机制
    #   phi = 1为SE
    #   phi = 2为CBAM
    #   phi = 3为ECA
    #-------------------------------#
    phi = 2
    cuda=True
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    input_shape=(416,416)
    anchors_path="data/yolo_anchors.txt"
    classes_path="data/yolo_classes.txt"
    # ------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    # ------------------------------------------------------#
    mosaic = True
    Cosine_lr=True
    smooth_label=0

    class_names=get_classes(classes_path)
    anchors=get_anchors(anchors_path)
    num_anchors=len(anchors)
    num_classes=len(class_names)

    model=Yolobody(num_anchors,num_classes,phi)
    weights_init(model)
    model_path="weight/yolov4_tiny_weights_voc_CBAM.pth"
    print("加载权重")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict=model.state_dict()#模型的参数字典
