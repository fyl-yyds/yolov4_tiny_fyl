import torch
import torch.nn as nn
import numpy as np
import math

class YOLOloss(nn.Module):
    def __init__(self,anchors,num_classes,image_size,smooth_label,cuda,normalize):
        super(YOLOloss, self).__init__()
        self.anchors=anchors
        self.num_anchors=len(anchors)
        self.num_classes=num_classes
        self.total=5+num_classes
        self.img_size=image_size #(W,H)
        self.feature_length=[image_size[0]//32,image_size[0]//16]#13份，26份
        self.smooth_label=smooth_label

        self.ignore_threshold=0.5
        self.lambda_conf=1.0
        self.lambda_cls=1.0
        self.lambda_loc=1.0
        self.cuda=cuda
        self.normalize=normalize

    def forward(self,predict,targets=None):
        #predict_shape:[N,3*(5+num_classes),13,13]/[N,3*(5+num_classes),26,26]
        N=predict.size(0)
        predict_h=predict.size(2)
        predict_w=predict.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------------------------------#
        stride_h=self.img_size[1]/predict_h
        stride_w=self.img_size[0]/predict_w
        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        scaled_anchors=[(a_w/stride_w,a_h/stride_h) for a_w,a_h in self.anchors]
        predicition=predict.reshape(N,int(self.num_anchors/2),self.total,
                    predict_h,predict_w).permute(0,1,3,4,2).contiguous()
        pre_conf=torch.sigmoid(predicition[...,4])
        pre_cls=torch.sigmoid(predicition[...,5:])
        #---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask               N, 3, predict_h, predict_w               有目标的特征点             [N,3,13,13]
        #   noobj_mask         N, 3, predict_h, predict_w               无目标的特征点             [N,3,13,13]
        #   target_box         N, 3, predict_h, predict_w, 4            中心宽高相对于特征层的值
        #   target_conf        N, 3, predict_h, predict_w               置信度真实值
        #   target_cls         N, 3, predict_h, predict_w, num_classes  种类真实值
        #   box_loss_x         N, 3, predict_h, predict_w               宽的真实值                值在[0~1之间]
        #----------------------------------------------------------------#
        mask,noobj_mask,target_box,target_conf,target_cls,box_loss_scale_x,box_loss_scale_y=self.get_target(
            targets,scaled_anchors,predict_w,predict_h)
        # ---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        # ----------------------------------------------------------------#
        noobj_mask,pred_boxes_for_ciou=self.get_ignore(predicition,targets,scaled_anchors,
            predict_w,predict_h,noobj_mask)#ignore:忽视

        if self.cuda:
            mask,noobj_mask         = mask.cuda(),noobj_mask.cuda()
            box_loss_scale_x,box_loss_scale_y   = box_loss_scale_x.cuda(),box_loss_scale_y.cuda()
            target_conf,target_cls  = target_conf.cuda(),target_cls.cuda()
            pred_boxes_for_ciou     = pred_boxes_for_ciou.cuda()
            target_box              = target_box.cuda()
        box_loss_scale=2-box_loss_scale_x*box_loss_scale_y
        # ---------------------------------------------------------------#
        #   计算预测结果和真实结果的CIOU
        # ----------------------------------------------------------------#
        ciou=(1-box_ciou(pred_boxes_for_ciou[mask.bool()],target_box[mask.bool()]))*box_loss_scale[mask.bool()]
        loss_loc=torch.sum(ciou)#https://blog.csdn.net/weixin_44791964/article/details/106214657
        # 计算置信度的loss
        loss_conf=torch.sum(BCELoss(pre_conf,mask)*mask)+torch.sum(BCELoss(pre_conf,mask)*noobj_mask)
        loss_cls=torch.sum(BCELoss(pre_cls[mask==1],smooth_labels(target_cls[mask==1],self.smooth_label,self.num_classes)))
        loss=loss_conf*self.lambda_conf+loss_cls*self.lambda_cls+loss_loc*self.lambda_loc

        if self.normalize:
            num_pos=torch.sum(mask)
            num_pos=torch.max(num_pos,torch.ones_like(num_pos))
        else:
            num_pos=N/2
        return loss,num_pos




    def get_target(self,targets,scaled_anchors,predict_w,predict_h):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        N=len(targets)  #targets:#0-255之间的值,ndarray(N,5),x中心点,y中心点,w,h,cls
        #-------------------------------------------------------#
        # 获得当前特征层先验框所属的编号，方便后面对先验框筛选
        # 13:[[3,4,5],[1,2,3]][0]->[3,4,5]   /  26: [[3,4,5],[1,2,3]][1]->[1,2,3]
        #-------------------------------------------------------#
        anchors_index=[[3,4,5],[1,2,3]][self.feature_length.index(predict_w)]
        #-------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        #-------------------------------------------------------#
        mask            = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        noobj_mask      = torch.ones (N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_x        = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_y        = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_w        = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_h        = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_box      = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,4,requires_grad=False)
        target_conf     = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        target_cls      = torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,self.num_classes,requires_grad=False)
        box_loss_scale_x= torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        box_loss_scale_y= torch.zeros(N,int(self.num_anchors/2),predict_h,predict_w,requires_grad=False)
        for n in range(N):
            if len(targets[n])==0:
                continue
            gxs=targets[n][:,0:1]*predict_w#doloader中的targets的值除了长宽，这里*13/26，相当于真是真实坐标除以32/16
            gys=targets[n][:,1:2]*predict_h#
            gws=targets[n][:,2:3]*predict_w#
            ghs=targets[n][:,3:4]*predict_h#不降维的取索引,保持二维形状[n,1]
            gxs_floor=torch.floor(gxs)#向下取整
            gys_floor=torch.floor(gys)
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            gt_box=torch.FloatTensor(torch.cat((torch.zeros_like(gws),torch.zeros_like(ghs),
                                               gws,ghs),dim=1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   6, 4
            #-------------------------------------------------------#
            anchors_box=torch.FloatTensor(torch.cat((torch.zeros(self.num_anchors,2),
                                                torch.FloatTensor(scaled_anchors)),dim=1))
            anch_iou=targets_anchors_iou(gt_box,anchors_box)#[n,6]
            #-------------------------------------------------------#
            #   计算重合度最大的先验框是哪个
            #   num_true_box,
            #-------------------------------------------------------#
            best_iou_index=torch.argmax(anch_iou,dim=-1)#一维数组
            for i,index in enumerate(best_iou_index):
                if index not in anchors_index:
                    continue
                # -------------------------------------------------------------#
                #   取出各类坐标：(相对于特征层)
                #   gx_floor和gy_floor代表的是真实框对应的特征点的x轴y轴坐标 整数
                #   gx和gy代表真实框的x轴和y轴坐标 整数+小数
                #   gw和gh代表真实框的宽和高
                # -------------------------------------------------------------#
                gx_floor=gxs_floor[i].long()
                gy_floor=gys_floor[i].long()
                gx=gxs[i]
                gy=gys[i]
                gw=gws[i]
                gh=ghs[i]
                if (gy_floor<predict_h) and (gx_floor<predict_w):
                    index =anchors_index.index(index)#[3,4,5].index(index) 返回索引
                    # ----------------------------------------#
                    #   noobj_mask代表无目标的特征点   noobj_mask[N,3,13,13]的1矩阵->noobj_mask[b, best_n, gj, gi]=0
                    # ----------------------------------------#
                    noobj_mask[n,index,gy_floor,gx_floor]=0
                    # ----------------------------------------#
                    #   mask代表有目标的特征点  mask[N,3,13,13]的0矩阵->mask[b, best_n, gj, gi] = 1
                    # ----------------------------------------#
                    mask[n,index,gy_floor,gx_floor]=1
                    # ----------------------------------------#
                    #   tx、ty代表中心的真实值    tx[N,3,13,13]的0矩阵->tx[b, best_n, gj, gi] = gx 小数+整数
                    # ----------------------------------------#
                    target_x[n,index,gy_floor,gx_floor]=gx
                    target_y[n,index,gy_floor,gx_floor]=gy
                    # ----------------------------------------#
                    #   tw、th代表宽高的真实值  tw[N,3,13,13]的0矩阵
                    # ----------------------------------------#
                    target_w[n,index,gy_floor,gx_floor]=gw
                    target_h[n,index,gy_floor,gx_floor]=gh
                    # ----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    # ----------------------------------------#
                    box_loss_scale_x[n,index,gy_floor,gx_floor]=targets[n][i,2] #[N,3,13,13]的0矩阵->真实框的宽(不是相对于特征层)
                    box_loss_scale_y[n,index,gy_floor,gx_floor]=targets[n][i,3] #[N,3,13,13]的0矩阵->真实框的高(值在[0~1]之间)
                    # ----------------------------------------#
                    #   tconf代表物体置信度
                    # ----------------------------------------#
                    target_conf[n,index,gy_floor,gx_floor]=1
                    # ----------------------------------------#
                    #   tcls代表种类置信度
                    # ----------------------------------------#
                    target_cls[n,index,gy_floor,gx_floor,targets[n][i,4].long()]=1  #[N,3,13,13,5]的0矩阵->标签的种类置信度cls
                else:
                    print(f"第{n}个标签超出界限")
                    print(f"gy_floor:在Y方向第{gy_floor}个格子,gx_floor:在X方向第{gx_floor}个格子,Y方向高度：{predict_h},X方向宽度{predict_w}")
                    continue
        target_box[...,0]=target_x #t_box [N,3,13,13,4]///X中心点的坐标 （小数+整数）
        target_box[...,1]=target_y #Y中心点的坐标 （小数+整数）
        target_box[...,2]=target_w #w
        target_box[...,3]=target_h #h
        return mask,noobj_mask,target_box,target_conf,target_cls,box_loss_scale_x,box_loss_scale_y
        #有目标的特征点；无目标的特征点；[中心点,宽高相对于特征层的值]，置信度N,3,13,13，类别置信度，N,3,13,13,5；真实的宽，高[N,3,13,13]
    def get_ignore(self,predicition,targets,scaled_anchors,predict_w,predict_h,noobj_mask):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        N=len(targets)
        #[N,30,13,13]的特征层对应的是面积最大的三个先验框，[N,30,26,26]对应的是面积小的三个先验框，原作者没有使用第一个框，重复的使用了第四个框
        anchors_index=[[3,4,5],[1,2,3]][self.feature_length.index(predict_w)]
        scaled_anchors=np.array(scaled_anchors)[anchors_index]

        x=torch.sigmoid(predicition[...,0]) #落在某个特征框中的点，相对于单个框的位置
        y=torch.sigmoid(predicition[...,1])
        w=predicition[...,2]
        h=predicition[...,3]

        FloatTensor=torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor=torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        #->[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.]->形状：[13(h),13(w)]->形状：[3,13(h),13(w)]->[N,3,13,13]
        grid_x=torch.linspace(0,predict_w-1,predict_w).repeat(predict_h,1).repeat(
            int(N*self.num_anchors/2),1,1).reshape(x.shape).type(FloatTensor)
        grid_y=torch.linspace(0,predict_h-1,predict_h).repeat(predict_w,1).repeat(
            int(N*self.num_anchors/2),1,1).reshape(y.shape).type(FloatTensor)
        # 取出三个相对于特征层大小的先验框的宽高[3,1]  #index_select:第一个数字表示维度，第二个数字表示该维度的第几个
        anchors_w=FloatTensor(scaled_anchors).index_select(1,LongTensor([0]))
        anchors_h=FloatTensor(scaled_anchors).index_select(1,LongTensor([1]))
        #[N,3,13,13]
        anchors_w=anchors_w.repeat(N,1).repeat(1,1,predict_h*predict_w).reshape(w.shape)
        anchors_h=anchors_h.repeat(N,1).repeat(1,1,predict_h*predict_w).reshape(h.shape)

        pred_boxes=FloatTensor(predicition[...,:4].shape)#[N,3,13,13,4]
        pred_boxes[...,0]=grid_x+x #(整数)+(小数)
        pred_boxes[...,1]=grid_y+y
        pred_boxes[...,2]=torch.exp(w)*anchors_w#
        pred_boxes[...,3]=torch.exp(h)*anchors_h
        for i in range(N):
            pred_box_ignore=pred_boxes[i]#[3,13,13,4]
            pred_box_ignore=pred_box_ignore.reshape(-1,4)#[3*169,4]
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(targets[i])>0:
                gx=targets[i][:,0:1]*predict_w
                gy=targets[i][:,1:2]*predict_h
                gw=targets[i][:,2:3]*predict_w
                gh=targets[i][:,3:4]*predict_h
                gt_box=torch.FloatTensor(torch.cat([gx,gy,gw,gh],-1)).type(FloatTensor)#[n,4]
                anchors_iou=targets_anchors_iou(gt_box,pred_box_ignore)#[n,4];[3*169,4] ->[n,3*169]
                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anchors_iou_max,_=torch.max(anchors_iou,dim=0)#返回[3*169]3*169个值和[索引]
                anchors_iou_max=anchors_iou_max.reshape(pred_boxes[i].size()[:3])#[3,13,13,4]
                noobj_mask[i][anchors_iou_max>self.ignore_threshold]=0
        return noobj_mask,pred_boxes#[N,3,13,13],[N,3,13,13,4]

def targets_anchors_iou(gt_box,anchors_box):
    b1_x1,b1_x2=gt_box[:,0]-gt_box[:,2]/2,gt_box[:,0]+gt_box[:,2]/2
    b1_y1,b1_y2=gt_box[:,1]-gt_box[:,3]/2,gt_box[:,1]+gt_box[:,3]/2
    b2_x1,b2_x2=anchors_box[:,0]-anchors_box[:,2]/2,anchors_box[:,0]+anchors_box[:,2]/2
    b2_y1,b2_y2=anchors_box[:,1]-anchors_box[:,3]/2,anchors_box[:,1]+anchors_box[:,3]/2
    box_1=torch.zeros_like(gt_box)
    box_2=torch.zeros_like(anchors_box)
    box_1[:,0],box_1[:,1],box_1[:,2],box_1[:,3]=b1_x1,b1_y1,b1_x2,b1_y2
    box_2[:,0],box_2[:,1],box_2[:,2],box_2[:,3]=b2_x1,b2_y1,b2_x2,b2_y2
    A=box_1.size(0)#n
    B=box_2.size(0)#6
    max_xy=torch.min(box_1[:,2:].unsqueeze(1).expand(A,B,2),# -> n,1,2 -> n,6,2 (通过复制)
                     box_2[:,2:].unsqueeze(0).expand(A,B,2))# -> 1,6,2 -> n,6,2 (通过复制) 每一行对比，取值正值中小的每行
    min_xy=torch.max(box_1[:,:2].unsqueeze(1).expand(A,B,2),
                     box_2[:,:2].unsqueeze(0).expand(A,B,2))
    inter=torch.clamp((max_xy-min_xy),min=0)#将(max_xy - min_xy)中的值保持在最小值和最大值内，确保宽高大于0

    inter=inter[:,:,0]*inter[:,:,1]#[n,6]交集

    area_1=((box_1[:,2]-box_1[:,0])*(box_1[:,3]-box_1[:,1])).unsqueeze(1).expand_as(inter)
    area_2=((box_2[:,2]-box_2[:,0])*(box_2[:,3]-box_2[:,1])).unsqueeze(0).expand_as(inter)

    all_area=area_1+area_2#并集
    iou=inter/(all_area-inter)
    return iou   # [n,6]

def box_ciou(b1,b2):#pred_boxes_for_ciou[mask.bool()],target_box[mask.bool()])
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #求出预测框左上角右下角
    b1_xy=b1[...,:2]
    b1_wh=b1[...,2:4]
    b1_wh_half=b1_wh/2
    b1_mins=b1_xy-b1_wh_half
    b1_maxes=b1_xy+b1_wh_half
    # 求出真实框左上角右下角
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half
    # 求真实框和预测框所有的iou
    intersect_mins=torch.max(b1_mins,b2_mins)
    intersect_maxes=torch.min(b1_maxes,b2_maxes)
    intersect_wh=torch.max(intersect_maxes-intersect_mins,torch.zeros_like(intersect_maxes))
    intersect_area=intersect_wh[...,0]*intersect_wh[...,1]
    b1_area=b1_wh[...,0]*b1_wh[...,1]
    b2_area=b2_wh[...,0]*b2_wh[...,1]
    union_area=b1_area+b2_area-intersect_area
    iou=intersect_area/torch.clamp(union_area,min=1e-6)
    #计算中心的差距
    center_distance=torch.sum(torch.pow((b1_xy-b2_xy),2),dim=-1)
    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins=torch.min(b1_mins,b2_mins)
    enclose_maxes=torch.max(b1_maxes,b2_maxes)
    enclose_wh=torch.max(enclose_maxes-enclose_mins,torch.zeros_like(intersect_maxes))
    # 计算对角线距离
    enclose_diagnoal=torch.sum(torch.pow(enclose_wh,2),dim=-1)
    ciou=iou-1.0*(center_distance)/torch.clamp(enclose_diagnoal,min=1e-6)

    v=(4/(math.pi**2))*torch.pow((torch.atan(b1_wh[...,0]/torch.clamp(b1_wh[...,1],min=1e-6))-
                                  torch.atan(b2_wh[...,0]/torch.clamp(b2_wh[...,1],min=1e-6))),2)
    alpha=v/torch.clamp((1.0-iou+v),min=1e-6)
    ciou=ciou-alpha*v
    return ciou

def BCELoss(pred,target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output
def clip_by_tensor(t,t_min,t_max):#将sigmoid之后的值限制在[t_min-7,t_max]之间
    t=t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
def smooth_labels(target_cls,smooth_label,num_classes):#target_cls[mask==1],self.smooth_label,self.num_classes
    return target_cls*(1.0-smooth_label)+smooth_label/num_classes

# a=torch.zeros(5,4)
# print(a.size)
# print(a.size())
# print(a.size(0))
# b=a[:,1:2]
# print(b.shape)
# print(torch.zeros_like(b))
# print(torch.cat([torch.zeros_like(b),b,],1))
# a=torch.arange(16).reshape(4,4)
# aa=torch.argmax(a,1)
# print(aa.shape)
# a=torch.linspace(0,13-1,13)
# print(a)
# print(a.shape)
# a=torch.arange(12).reshape(3,4)
# print(torch.sum(a,dim=-1))
# print(torch.sum(a,axis=-1))
# a=torch.tensor([[56,26, 56,87, 85,120],[1,2,3,4,5,6]])
# print(a.index_select(0,torch.LongTensor([1])))
# print([3,4,5].index(3))