import xml.etree.ElementTree as ET
import glob
import numpy as np

def load_data(xml_path):
    data=[]
    for xml in glob.glob(f"{xml_path}/*xml"):
        #glob.glob:返回所有匹配的文件路径列表
        tree=ET.parse(xml)
        height=float(tree.findtext("size/height"))
        width=float(tree.findtext("size/width"))
        if height<=0 or width<=0:
            continue

        for obj in tree.iter("item"):
            # 将标签限制到图片宽高范围内
            xmin=max(int(float(obj.findtext("bndbox/xmin")))/width,0.0)
            ymin=max(int(float(obj.findtext("bndbox/ymin")))/height,0.0)
            xmax=max(int(float(obj.findtext("bndbox/xmax")))/width,0.0)
            ymax=max(int(float(obj.findtext("bndbox/ymax")))/height,0.0)

            xmin = np.float64(min(xmin, width))
            xmax = np.float64(min(xmax, width))
            ymin = np.float64(min(ymin, height))
            ymax = np.float64(min(ymax, height))
            # 得到宽高
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)

def kmeans(xml_w_h,anchors_num):
    # 取出一共有多少目标框
    box_num=xml_w_h.shape[0]

    # 每个框各个点的位置
    n_6=np.empty((box_num,anchors_num))
    ##返回指定形状的由随机数组成的数组，数组元素不为空

    # 最后的聚类位置
    n_=np.zeros((box_num,))#形状是[n,] 表示以为数组[1,n]

    np.random.seed()
    # 随机选6个当聚类中心
    random_6=xml_w_h[np.random.choice(box_num,anchors_num,replace=False)]
    ##从比(box_num)小的数中随机取出anchors_num个数，[0,box_num-1]，replace=False表示不能取到重复的值
    while True:
        for i in range(box_num):
            # 计算每一行距离六个点的iou情况。
            n_6[i]=1-cas_iou(xml_w_h[i],random_6)
        # 取出最小点
        n_6_min=np.argmin(n_6,axis=1)#给出对应维度方向最小值的下标；[n,]

        if (n_==n_6_min).all():#n_完全等于n_6_min,最小值全在第一列
            break

        #求每一个类的均值点
        for j in range(anchors_num):
            random_6[j]=np.median(xml_w_h[n_6_min==j],axis=0)
        n_=n_6_min
    return random_6#6个框

def cas_iou(xml_w_h_1,random_6):
    x=np.minimum(random_6[:,0],xml_w_h_1[0])
    y=np.minimum(random_6[:,1],xml_w_h_1[1])

    intersection=x*y #交叉
    xml_area=xml_w_h_1[0]*xml_w_h_1[1]
    num_6_area=random_6[:,0]*random_6[:,1]

    iou=intersection/(xml_area+num_6_area-intersection)
    return iou #返回的是1维数组[6,]
    #当intersection=random_6的时候，iou最大，所以随机的6个框和每个目标框宽高越接近，iou越高，1-iou越小
def avg_iou(xml_w_h,out):
    # print([np.max(cas_iou(xml_w_h[i],out)) for i in range(xml_w_h.shape[0])])
    # print(len([np.max(cas_iou(xml_w_h[i],out)) for i in range(xml_w_h.shape[0])]))
    # print(np.mean([np.max(cas_iou(xml_w_h[i],out)) for i in range(xml_w_h.shape[0])]))
    return np.mean([np.max(cas_iou(xml_w_h[i],out)) for i in range(xml_w_h.shape[0])])

if __name__ == '__main__':
    # 运行该程序会计算"\VOC2021\Annotation"的xml
    # 会生成yolo_anchors.txt
    path=r"E:\666666\data\7-6\data.7.9"
    size=416
    anchors_num=6
    xml_path=path+r"\VOC2021\Annotations"
    #取出所有xml中的所有框
    xml_w_h=load_data(xml_path)

    out=kmeans(xml_w_h,anchors_num)
    out_area=out[:,0]*out[:,1]
    out=out[np.argsort(out_area)]#argsort:按从小到大排列，返回索引
    print("acc:{:.2f}%".format(avg_iou(xml_w_h,out)*100))
    data=out*size
    print(data)
    f=open("data/yolo_anchors.txt","w")
    row=np.shape(data)[0]#np.shape返回参数的形状
    for i in range(row):
        if i ==0:
            x_y="%d,%d" % (data[i][0],data[i][1])
        else:
            x_y=", %d,%d" % (data[i][0],data[i][1])
        f.write(x_y)
    f.close()


    # print(np.random.randn()) #每次都不一样
    # np.random.seed(0)#加了这个，下面这行每次随机数都和第一次一样
    # print(np.random.rand())

    a=np.arange(8).reshape(4,2)
    # print(a)
    # print(a[:,1]*a[:,0])
    # print((a[:,1]*a[:,0]).shape)
    # import torch
    # print(torch.tensor(a[:,1]*a[:,0]).unsqueeze(0).shape)
    # print(np.minimum(a[:,1],5))
    # print(np.argmin(a,axis=1))
    # print(np.argmin(a,axis=1).shape)
    # a=np.zeros((6,2))
    # b=np.array([1,3])
    # print(cas_iou(b,a).shape)
    # a=np.array([0,0,1,1])
    # b=np.array([1,0,1,1])
    # print(a==b)
    # print((a==b).all())
    # print(a==1)
    print(np.max(np.array([1,2,3,3,2,1])))