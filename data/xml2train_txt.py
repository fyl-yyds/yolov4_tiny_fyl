import os
import random

random.seed(0) #保证每次的随机数相同
xml_path=r"E:\666666\data\7-6\data.7.9\VOC2007\Annotations"

train_val_persent=1
train_persent=0.9

all_xml=os.listdir(xml_path)
xml_list=[]
for xml in all_xml:
    if xml.endswith(".xml"):
        xml_list.append(xml)

num=len(xml_list)
train_val_num=random.sample(range(num),int(train_val_persent*num))#从num中随机取出int(train_val_persent*num)个数
train_num=random.sample(range(num),int(train_persent*num))
print(f"验证集+训练集的数量：{int(train_val_persent*num)}")
print(f"训练集的数量：{int(train_persent*num)}")

train_val=open("num/train_val.txt", "w")
train=open("num/train.txt", "w")
val=open("num/val.txt", "w")
test=open("num/test.txt", "w")

for i in range(num):
    name=xml_list[i][:-4]+"\n"
    if i in train_val_num:
        train_val.write(name)
        if i in train_num:
            train.write(name)
        else:
            val.write(name)
    else:
        test.write(name)

train_val.close()
train.close()
val.close()
test.close()
