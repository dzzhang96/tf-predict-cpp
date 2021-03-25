# tf-predict-cpp

Use Tensorflow C++ API and the pre-trained model to segment the images. The input can either be a single image or a stacked image data, depending on the network you choose. In this project, two examples of U-net and V-net are shown.

### Environment: 

The project requires OpenCV and Tensorflow C++ 1.10 library.You can compile the tenseorflow C++ library by yourself or download it from [here](https://github.com/fo40225/tensorflow-windows-wheel). But please note that there might be a difference for Tensorflow 2.0+.   

### How to use

#### 1. Train and save your model

Here we trained U-net for segmenting the bone graft in maxillary sinus and V-net to segment the mandible.

#### 2. Freeze it

Freeze your model from .meta to .pb using "freeze_model.py". Be sure to find your correct output node name.

#### 3. Predict

The correspoding size of the image and the layer of your network are to be modified on your need. For the latter part, I recommand [netron](https://github.com/lutzroeder/netron) to find the name. The command arguments in visual studio shall be: your model path and your images for prediction. 

In this project: 

U-net: ..\model\BG-Segment-unet.pb ..\test\Img8_235.jpg

![](https://github.com/dzzhang96/tf-predict-cpp/blob/master/test/Img8_235.JPEG)![](https://github.com/dzzhang96/tf-predict-cpp/blob/master/test/Img8_235.jpg)

V-net: ..\model\Vnet3dModule.pb ..\test\0.jpg ..\test\1.jpg ..\test\2.jpg ..\test\3.jpg ..\test\4.jpg ..\test\5.jpg ..\test\6.jpg ..\test\7.jpg ..\test\8.jpg ..\test\9.jpg ..\test\10.jpg ..\test\11.jpg ..\test\12.jpg ..\test\13.jpg ..\test\14.jpg ..\test\15.jpg

![](https://github.com/dzzhang96/tf-predict-cpp/blob/master/test/vnet.jpg)








