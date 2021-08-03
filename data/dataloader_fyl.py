import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import cv2


class Yolodataset(Dataset):
    def __init__(self,train_lines,image_size,mosaic=True,random=True):
        super(Dataset, self).__init__()
        self.train_lines=train_lines
        self.train_num=len(train_lines)
        self.image_size=image_size
        self.mosaic=mosaic
        self.true_false=True
        self.random=random
    def __len__(self):
        return self.train_num

    def rand(self,a=0,b=1):
        return np.random.rand()*(b-a)+a
    #np.random.rand()的值在[0,1]之间
    #自定义的rand函数：随机生成(a,b)之间的一个值

    def get_random_data(self,every_line,input_shape,ratio1=0.3,hue=0.1,sat=1.5,val=1.5,random=True):
        img_path=every_line.split()
        image=Image.open(img_path[0])
        image_w,image_h=image.size  #PIL打开格式是W，H
        input_h,input_w=input_shape
        box=np.array([np.array(list(map(int,box.split(",")))) for box in img_path[1:]])
        #返回的是[n,5]的二位数组


        if not random:#(不随机的话)将图片进行加灰边的resize
            scale=min(input_w/image_w,input_h/image_h)#取最小比列
            new_w=int(image_w*scale)
            new_h=int(image_h*scale)
            dx=(input_w-new_w)//2  #获得灰边的距离
            dy=(input_h-new_h)//2

            image=image.resize((new_w,new_h),Image.BICUBIC)
            new_image=Image.new("RGB",(input_w,input_h),(128,128,128))
            new_image.paste(image,(dx,dy))
            image_array=np.array(new_image,np.float32)

            # 调整目标框坐标(将目标框resize到等比缩放的图片中)
            new_box=np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*(new_w/input_w)+dx
                box[:,[1,3]]=box[:,[1,3]]*(new_h/input_h)+dy
                box[:,0:2][box[:,0:2]<0]=0
                box[:,2][box[:,2]>input_w]=input_w
                box[:,3][box[:,3]>input_h]=input_h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h>1)]#保留有效框
                # new_box=np.zeros(len(box),5)
                # new_box[:,len(box)]=box
                new_box=box
            return image_array,new_box

        if random:
            # 调整图片大小  ratio1=0.3 在input_w=input_h的情况下new_ar=[0.53,1.85]
            new_ratio=input_w/input_h*self.rand(1-ratio1,1+ratio1)/self.rand(1-ratio1,1+ratio1)
            scale=self.rand(0.25,2)
            if new_ratio <1:
                new_h=int(scale*input_h)
                new_w=int(new_h*new_ratio)
            else:
                new_w=int(scale*input_w)
                new_h=int(new_w/new_ratio)
            image=image.resize((new_w,new_h),Image.BICUBIC)

            dx=int(self.rand(0,input_w-new_w))
            dy=int(self.rand(0,input_h-new_h))
            new_image=Image.new("RGB",(input_w,input_h),
                (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
            new_image.paste(image,(dx,dy))
            image=new_image#随机缩放粘贴，背景颜色随机

            #随机左右翻转0.5的概率
            flip=self.rand()<.5
            if flip:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)

            # 色域变换hue=.1, sat=1.5, val=1.5
            hue=self.rand(-hue,hue)
            sat=self.rand(1,sat) if self.rand()<.5 else 1/self.rand(1,sat)
            val=self.rand(1,val) if self.rand()<.5 else 1/self.rand(1,val)
            x=cv2.cvtColor(np.array(image,np.float32)/255,cv2.COLOR_RGB2HSV)
            x[...,0]+=hue*360
            x[...,0][x[...,0]>1]-=1
            x[...,0][x[...,0]<0]+=1    #色调H：用角度度量，取值范围为0°～360°
            x[...,1]*=sat              #饱和度S：取值范围为0.0～1.0；
            x[...,2]*=val              #亮度V：取值范围为0.0(黑色)～1.0(白色)
            x[x[...,0]>360,0]=360
            x[...,1:][x[...,1:]>1]=1
            x[x<0]=0
            img_array=cv2.cvtColor(x,cv2.COLOR_HSV2RGB)*255

            # 调整目标框坐标
            new_box=np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*(new_w/image_w)+dx
                box[:,[1,3]]=box[:,[1,3]]*(new_h/image_h)+dy
                if flip:
                    box[:, [0, 2]] = input_w - box[:, [2, 0]]
                    for _box in box:  # 镜像图片时，将左右手标签互换
                        if _box[4] == 3:
                            _box[4] = 4
                        elif _box[4] == 4:
                            _box[4] = 3

                box[:,0:2][box[:,0:2]<0]=0
                box[:, 2][box[:, 2] > input_w] = input_w
                box[:, 3][box[:, 3] > input_h] = input_h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h>1)]
                # new_box=np.zeros((len(box),5))
                # new_box[:len(box)]=box
                new_box=box
            return img_array,new_box

    def get_random_data_mosaic(self,every_line,input_shape,ratio1=0.3,hue=0.1,sat=1.5,val=1.5):
        input_h,input_w=input_shape
        min_offset_x=0.3
        min_offset_y=0.3
        scale_low=1-min(min_offset_x,min_offset_y)
        scale_high=scale_low+0.2#这里可以修改

        image_datas=[]
        box_datas=[]
        index=0
        # 缩小图片后，将图片贴在哪个位置(x,y)
        place_x=[0,0,int(input_w*min_offset_x),int(input_w*min_offset_x)]
        place_y=[0,int(input_h*min_offset_y),int(input_h*min_offset_y),0]
        for line in every_line:
            line=line.split()
            image=Image.open(line[0])
            image=image.convert("RGB")
            image_w,image_h=image.size
            box=np.array([np.array(list(map(int,box.split(",")))) for box in line[1:]])

            flip=self.rand()<.5
            if flip and len(box)>0:
                image=image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:,[0,2]]=image_w-box[:,[2,0]]
                # 左右手标签互换互换
                for _box in box:
                    if _box[4] == 3:
                        _box[4] = 4
                    elif _box[4] == 4:
                        _box[4] = 3

            new_ratio=input_w/input_h
            scale =self.rand(scale_low,scale_high)
            if new_ratio<1:
                new_h=int(scale*input_h)
                new_w=int(new_h*new_ratio)
            else:
                new_w=int(scale*input_w)
                new_h=int(new_w/new_ratio)
            image=image.resize((new_w,new_h),Image.BICUBIC)

            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
            image=Image.fromarray((image*255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置,粘贴到输入尺寸的底图上
            dx=place_x[index]
            dy=place_y[index]
            new_image=Image.new("RGB",(input_w,input_h),
                (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)))
            new_image.paste(image,(dx,dy))
            image_array=np.array(new_image)
            index+=1

            new_box=[]
            if len(box)>0:
                np.random.shuffle(box)
                box[:,[0,2]]=box[:,[0,2]]*(new_w/image_w)+dx
                box[:,[1,3]]=box[:,[1,3]]*(new_h/image_h)+dy
                box[:,:2][box[:,:2]<0]=0
                box[:,2][box[:,2]>input_w]=input_w
                box[:,3][box[:,3]>input_h]=input_h
                box_w=box[:,2]-box[:,0]
                box_h=box[:,3]-box[:,1]
                box=box[np.logical_and(box_w>1,box_h>1)]
                new_box=np.zeros((len(box),5))
                new_box[:len(box)]=box
            image_datas.append(image_array)
            box_datas.append(new_box)
        # 将图片分割，放在一起，现在开始正式分割图片，从（cutx，cuty）分开。
        cutx=np.random.randint(int(input_w*min_offset_x),int(input_w*(1-min_offset_x)))
        cuty=np.random.randint(int(input_h*min_offset_y),int(input_h*(1-min_offset_y)))
        # 将四张图片拼接在一起
        new_image=np.zeros([input_h,input_w,3])#RGB图片格式是HWC
        new_image[:cuty,:cutx,:]=image_datas[0][:cuty,:cutx,:]
        new_image[cuty:,:cutx,:]=image_datas[1][cuty:,:cutx,:]
        new_image[cuty:,cutx:,:]=image_datas[2][cuty:,cutx:,:]
        new_image[:cuty,cutx:,:]=image_datas[3][:cuty,cutx:,:]
        # 对框进行进一步的处理
        new_boxes=np.array(merge_box_datas(box_datas,cutx,cuty))
        return new_image,new_boxes
        # 四张图片拼接出来的一张图片(416,416,3),四张图片所有的框，二维的ndarray

    def __getitem__(self, index):
        lines=self.train_lines
        num=self.train_num
        index=index%num
        if self.mosaic:
            if self.true_false and (index+4)<num:
                img,y=self.get_random_data_mosaic(lines[index:index+4],self.image_size[0:2])
            else:#不足四张图片
                img,y=self.get_random_data(lines[index],self.image_size[0:2],random=self.random)
            self.true_false=bool(1-self.true_false)
        else:
            img,y=self.get_random_data(lines[index],self.image_size[0:2],random=self.random)

        if len(y)!=0:
            # 从坐标转换成0~1的百分比
            boxes=np.array(y[:,:4],dtype=np.float32)
            boxes[:,0]=boxes[:,0]/self.image_size[1]
            boxes[:,1]=boxes[:,1]/self.image_size[0]
            boxes[:,2]=boxes[:,2]/self.image_size[1]
            boxes[:,3]=boxes[:,3]/self.image_size[0]

            boxes=np.maximum(np.minimum(boxes,1),0)
            boxes[:,2]=boxes[:,2]-boxes[:,0]#宽
            boxes[:,3]=boxes[:,3]-boxes[:,1]#高
            boxes[:,0]=(boxes[:,0]+boxes[:,2])/2
            boxes[:,1]=(boxes[:,1]+boxes[:,3])/2
            y=np.concatenate([boxes,y[:,-1:]],axis=-1)
            # 返回x、y中心点，w、h,类别
        img=np.array(img,dtype=np.float32)
        final_img=np.transpose(img/255.0,(2,0,1))#HWC->CHW
        final_targets=np.array(y,dtype=np.float32)
        return final_img,final_targets
#是否马赛克处理的图片(HWC)(416,416)//#0-255之间的值,ndarray(N,5),x中心点,y中心点,w,h,cls

# collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
def yolo_dataset_collate(batch):
    images=[]
    boxes=[]
    for img,box in batch:
        images.append(img)
        boxes.append(box)
    images=np.array(images)
    return images,boxes

def merge_box_datas(box_datas, cutx, cuty):
    merge_boxes = []
    for i in range(len(box_datas)):
        for box in box_datas[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:  # 第一张图片(左上角)
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:  # 第二张图片(左下角)
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:  # 第三张图片(右下角)
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:  # 第四张图片(右上角)
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_boxes.append(tmp_box)
    return merge_boxes
# if __name__ == '__main__':
    # with open(r"F:\python project\first_project\backbone\2007_val.txt") as f:
    #     lines=f.readlines()[0].split()
    #     print(lines)
    #     print(np.array([np.array(list(map(int,box.split(",")))) for box in lines[1:]]))

    # def rand(a=0,b=1):
    #     return np.random.rand()*(b-a)+a
    # print(rand(0.7,1.3))
    # print(rand(0.7,1.3))
    # im=Image.open("1.PNG")
    # print(im.size)