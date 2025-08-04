# RetinalCoNet
This repository provides code for running inference using the RetinalCoNet model, as well as example notebooks demonstrating how to use the model.

## RetinalCoNet: Underwater Fish Segmentation Network Based on Bionic Retinal Dual Channel and Multi-module Cooperation
### 

torch==2.6.0
torchvision==0.16.2+cu121

### Train the VOC dataset
 
1. Before training, place the label file in the SegmentationClass folder under the VOC2007 folder in the VOCdevkit folder.    
2. Before training, place the image files in the JPEGImages folder under the VOC2007 folder in the VOCdevkit folder.    
3. Use the voc_annotation.py file to generate the corresponding txt file before training.    
4. Note that you need to modify the num_classes parameter in train.py to the number of classes plus one.    
5. Run train.py to start training.  

### Prediction steps

1. Place the trained weight path in mask-generator.py.
2. Run predict.py to start detection.

### Evaluation steps
Run the eval.py file to obtain the evaluation indicators.

