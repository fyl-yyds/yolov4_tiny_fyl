import torch



class YOLO(object):
    _default={
        "model_path"        :"original_weight/yolov4_tiny_weights_voc_CBAM.pth",
        "anchors_path"      :"data/yolo_anchors.txt",
        "classes_path"      :"data/yolo_classes.txt",
        # -------------------------------#
        #   所使用的注意力机制的类型
        #   phi = 0为不使用注意力机制
        #   phi = 1为SE
        #   phi = 2为CBAM
        #   phi = 3为ECA
        # -------------------------------#
        "phi"               :2,
        "model_image_size"  :(416,416,3),
        "confidence"        :0.5,
        "iou"               :0.3,
        "cuda"              :True,

    }