# Semantic Segmentation Nets (SSNeT)
Semantic Segmentation is the process of taking an image and label every single pixel to it's corresponding class. We can think of semantic segmentation as image classification at a pixel level. For example, in an image that has many cars, segmentation will label all the objects as car objects. However, a separate class of models known as instance segmentation is able to label the separate instances where an object appears in an image.

![](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png)

In this project you will find some implementation of Deep Learning Models with examples for Semantic Segmentation.

## FCN8
![](https://miro.medium.com/max/790/1*wRkj6lsQ5ckExB5BoYkrZg.png)

Fully Convolutional Network (https://arxiv.org/abs/1605.06211) was released in 2016 and achieves a performance of 67.2% mean IU on PASCAL VOC 2012. FCNs uses ILSVRC classifiers which were casted into fully-convolutional networks and augmented for dense prediction using pixel-wise loss and in-network up-sampling. This means that previous classfier like VGG16, AlexNet or GoogleNet are used as Encoders and the Fully Connected Layers or Dense Layers are replaced by convolutions. Then, the downsampled features are upsampled by transpose convolutions. The upsample layers are initialized with bilinear interpolation but this upsample layers learns to upsample. Skip connection are used to combine fine layers and coarse layers, which lets the model to make local predictions that respect global structure.

You can find the original implementation at https://github.com/shelhamer/fcn.berkeleyvision.org, but it uses Caffe.

### Our Implementation
Our implementation uses a VGG16 network as Encoder. The main differences with the author's implementation are:
* We use of BatchNormalization by default (but it can be disabled)
* We don't use double learning rate for biases
* We use l2 regularization instead of weight decay
* We have support for depthwise separable convolutions

## Depthwise Separable FCN8 (FastFCN8)
Inspired by the recent success of Depthwise Separable Convolution we build the FastFCN8 model. Please, this FastFCN8 model IS NOT the model refered in this [paper](https://paperswithcode.com/paper/fastfcn-rethinking-dilated-convolution-in-the)
This FCN8 implementation include support for Depthwise separable convolution which allows the model to run faster and reduce drastically the memory usage without losing performance accuracy. The FastFCN8 model reduce the number of trainable parameters of the FCN8 from 134.278.854 to 20.607.073. The performance of the model is tested in the example notebooks provided.

To use standard FCN8 or FastFCN8 build the model changing the conv_type parameter from 'conv' to 'ds' respectively.

```
FCN8 = build_model(*params, conv_type='conv')
FastFCN8 = build_model(*params, conv_type='ds')

```
