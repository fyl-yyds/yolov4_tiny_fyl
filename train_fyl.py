import numpy as np
import torch
import torch.backends.cudnn as cudann
from nets.yolov4_tiny import Yolobody
from yolo_traing_fyl import YOLOloss
import os
from matplotlib import pyplot as plt
import scipy.signal
import torch.optim as optim
from data.dataloader_fyl import Yolodataset,yolo_dataset_collate
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("log2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []

        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Ferrze_ornot_Epoch,cuda):
    total_loss=0
    val_loss=0
    net.train()
    with tqdm(total=epoch_size,desc=f"Epoch{epoch+1}/{Ferrze_ornot_Epoch}",postfix=dict,mininterval=0.3) as pbar:
        for iteration,batch in enumerate(gen):
            if iteration>=epoch_size:#iteration(迭代)
                break
            images,targets=batch[0],batch[1]
            with torch.no_grad():
                if cuda:
                    images=torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]

            outputs=net(images)
            losses=[]
            num_pos_all=0
            for i in range(2):
                loss_item,num_pos=yolo_loss(outputs[i],targets)
                losses.append(loss_item)
                num_pos_all+=num_pos
            loss=sum(losses)/num_pos_all

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

            pbar.set_postfix(**{"total_loss":total_loss/(iteration+1),
                                "lr":get_lr(optimizer)})
            pbar.update(1)
    # 每一个epoch，记录各层权重、梯度
    for name, param in net.named_parameters():  # 返回网络的
        # writer.add_histogram(name + '_grad', param.grad, epoch)
        writer.add_histogram(name + '_data', param, epoch)
        writer.close()

    net.eval()
    print("开始验证")
    with tqdm(total=epoch_size_val,desc=f"Epoch{epoch+1}/{Ferrze_ornot_Epoch}",postfix=dict,mininterval=0.3) as pbar:
        for iteration,batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val,targets_val=batch[0],batch[1]
            with torch.no_grad():
                if cuda:
                    images_val=torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val=[torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs=net(images_val)
                losses=[]
                num_pos_all=0
                for i in range(2):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    loss_history.append_loss(total_loss/(epoch_size+1),val_loss/(epoch_size_val+1))
    print("验证结束")
    print("Epoch:"+str(epoch+1)+"/"+str(Ferrze_ornot_Epoch))
    print("Total loss: %.4f || Val loss: %.4f "% (total_loss/(epoch_size+1),val_loss/(epoch_size+1)))
    print("Saving state,iter；",str(epoch+1))
    torch.save(model.state_dict(),'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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
    input_shape=(416,416)   #(W,H)
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
    smooth_label=0.005

    class_names=get_classes(classes_path)
    anchors=get_anchors(anchors_path)
    num_anchors=len(anchors[0])
    num_classes=len(class_names)

    model=Yolobody(num_anchors,num_classes,phi)
    weights_init(model)
    model_path= "original_weight/yolov4_tiny_weights_voc_CBAM.pth" ##########################################################################################
    print("加载权重到state dict中...")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict=model.state_dict()#模型的参数字典
    pretrained_dict=torch.load(model_path,map_location=device)
    pretrained_dict={k:v for k,v in pretrained_dict.items() if np.shape(model_dict[k])==np.shape(v)}
    # items()返回字典中的键、值对元组
    model_dict.update(pretrained_dict)#更新
    model.load_state_dict(model_dict)
    print("加载结束！")

    net=model.train()
    if cuda:
        net=torch.nn.DataParallel(model) #可以用多个GPU训练YoloBody
        cudann.benchmark = True #Benchmark模式会提升计算速度
        net=net.cuda()

    yolo_loss=YOLOloss(np.reshape(anchors,[-1,2]),num_classes,
                       (input_shape[1],input_shape[0]),smooth_label,cuda,normalize)
    loss_history=LossHistory("logs/")

    annotation_path= "data/2007/train2007.txt"
    #----------------------------------------------------------------------#
    #   验证集的划分在train_fyl.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split=0.1
    with open(annotation_path) as f:
        lines=f.readlines()
    np.random.seed(10101)
    # seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，
    # 则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    np.random.shuffle(lines)#随机打乱一次数据,之后每次数据都和第一次打乱一样
    np.random.seed(None)
    num_val=int(len(lines)*val_split) #428  int()向下取整  len(lines)=4287
    num_train=len(lines)-num_val      #4287-428=3859
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        lr=1e-3
        Batch_size=50
        Init_Epoch=0
        Freeze_Epoch=50
        #----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        #----------------------------------------------------------------------------#
        optimizer=optim.Adam(net.parameters(),lr)
        if Cosine_lr:
            lr_scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5,eta_min=1e-5)
        else:
            lr_scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.92)
        train_dataset=Yolodataset(lines[:num_train],(input_shape[0],input_shape[1]),mosaic=mosaic,random=True)
        val_dataset=Yolodataset(lines[num_train:],(input_shape[0],input_shape[1]),mosaic=False,random=False)
        gen=DataLoader(train_dataset,shuffle=True,batch_size=Batch_size,num_workers=4,pin_memory=True,
                       drop_last=True,collate_fn=yolo_dataset_collate)#pin_memory:锁页内存，当计算机的内存充足的时候，可以设置pin_memory=True
        gen_val=DataLoader(val_dataset,shuffle=True,batch_size=Batch_size,num_workers=4,pin_memory=True,
                        drop_last=True,collate_fn=yolo_dataset_collate)
        epoch_size=num_train//Batch_size
        epoch_size_val=num_val//Batch_size
        if epoch_size==0 or epoch_size_val==0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad=False
        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_loss,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 30
        Freeze_Epoch = 50
        Unfreeze_Epoch = 700

        # ----------------------------------------------------------------------------#
        #   我在实际测试时，发现optimizer的weight_decay起到了反作用，
        #   所以去除掉了weight_decay，大家也可以开起来试试，一般是weight_decay=5e-4
        # ----------------------------------------------------------------------------#
        optimizer = optim.Adam(net.parameters(), lr)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = Yolodataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, random=True)
        val_dataset = Yolodataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, random=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, cuda)
            lr_scheduler.step()



###########################################################################################################