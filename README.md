# Computer Vision Nanodegree Udacity
## Image-Captioning-Project

## Project Overview

In this project, we will create a neural network architecture to automatically generate captions from images.<br>
After using the Microsoft Common Objects in COntext (MS COCO) dataset to train your network, we will test your network on novel images!

## Image Captioning Model
![](/images/image-captioning.png)


## CNN Encoder

I used same pretrained resnet50 from pytorch model zoo which was given to me which has been pretrained version of the network trained on more than a million images from the ImageNet database without its top layer as we know the initial layers can identify more general features , than i take the output of that last layer and map it to the linear layer which take last layer's output as a input and map it to embedded size i.e its output size = embed size.

![](images/encoder.png)



## LSTM Decoder

In the project, we pass all our inputs as a sequence to an LSTM. A sequence looks like this: first a feature vector that is extracted from an input image, then a start word, then the next word, the next word, and so on!<br>

The architecture and implementation of this is written in model.py file.

![](images/decoder.png)

## Combined Model

![](images/encoder-decoder.png)






















# Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  https://classroom.udacity.com/nanodegrees/nd891/parts/827d2ba4-b67b-4146-bf74-a88a87f860a9/modules/80408aa5-e4a9-4e66-b599-a282402ee8f1/lessons/44a921a7-031a-4d18-80f2-b28d80819cf3/concepts/7b052a2d-d45a-4101-ba64-bd76f5f11564#
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).
