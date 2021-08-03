import xml.etree.ElementTree as ET

classes=["脸","眼","嘴","手"]
data_path=r"E:\666666\data\7-6\data.7.9"
def add_xywhcls(year,name,cls_2021):
    xml=open(data_path+f"/VOC{year}/Annotations/{name}.xml",encoding="UTF-8")#读取xml加上UTF-8，不然读取会出错
    tree=ET.parse(xml)
    root=tree.getroot()
    h=int(tree.findtext("size/height"))
    w=int(tree.findtext("size/width"))
    for obj in root.iter("item"):
        cls1 =obj.find("name").text
        if cls1 not in classes:
            continue
        cls1_id=classes.index(cls1)
        xmlbox=obj.find("bndbox")
        b=[int(xmlbox.find("xmin").text),int(xmlbox.find("ymin").text),
           int(xmlbox.find("xmax").text),int(xmlbox.find("ymax").text)]

        for i in range(len(b)):
            b[i]=max(b[i],0)
            if i%2==0:
                b[i]=min(b[i],w)
            else:
                b[i]=min(b[i],h)
        cls_2021.write(" "+",".join(str(a) for a in b)+","+str(cls1_id))#",".join("abcd") 将,加在abcd中


year_cls=[("2007","train"),("2007","val"),("2007","test")]
for year,cls in year_cls:
    names=open(f"num\{cls}.txt",encoding="UTF-8").read().strip().split()
    cls_2007=open(f"2007\{cls}{year}.txt",encoding="UTF-8",mode="w")
    for name in names:
        cls_2007.write(f"{data_path}\VOC{year}\JPEGImages\{name}.jpg")
        add_xywhcls(year,name,cls_2007)
        cls_2007.write("\n")
    cls_2007.close()