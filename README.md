# RetinalCoNet
This repository provides code for running inference using the RetinalCoNet model, as well as example notebooks demonstrating how to use the model.

## RetinalCoNet: Underwater Fish Segmentation Network Based on Bionic Retinal Dual Channel and Multi-module Cooperation
---

### 

torch==2.6.0
torchvision==0.16.2+cu121

### 训练步骤
#### 一、训练voc数据集
 
1、训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的SegmentationClass中。    
2、训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。    
3、在训练前利用voc_annotation.py文件生成对应的txt。    
4、注意修改train.py的num_classes为分类个数+1。    
5、运行train.py即可开始训练。  


### 预测步骤

运行predict.py开始检测。    

```
#### 二、使用自己训练的权重
1. 按照训练步骤训练。    
2. 在unet.py文件里面，在如下部分修改model_path、backbone和num_classes使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件**。    
```python
_defaults = {
    #-------------------------------------------------------------------#
    #   model_path指向logs文件夹下的权值文件
    #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
    #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
    #-------------------------------------------------------------------#
    "model_path"    : 'model_data/unet_vgg_voc.pth',
    #--------------------------------#
    #   所需要区分的类的个数+1
    #--------------------------------#
    "num_classes"   : 21,
    #--------------------------------#
    #   所使用的的主干网络：vgg、resnet50   
    #--------------------------------#
    "backbone"      : "resnet50",
    #--------------------------------#
    #   输入图片的大小
    #--------------------------------#
    "input_shape"   : [512, 512],
    #--------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------#
    "cuda"          : True,
}
```

### 评估步骤
运行eval.py文件即可得到评价指标


