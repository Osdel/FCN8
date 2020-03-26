# Semantic Segmentation Nets (SSNeT)
Semantic Segmentation is the process of taking an image and label every single pixel to it's corresponding class. We can think of semantic segmentation as image classification at a pixel level. For example, in an image that has many cars, segmentation will label all the objects as car objects. However, a separate class of models known as instance segmentation is able to label the separate instances where an object appears in an image.

![](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png)

In this project you will find some implementation and examples of Deep Learning Models for Semantic Segmentation.

## FCN8
![](https://miro.medium.com/max/790/1*wRkj6lsQ5ckExB5BoYkrZg.png)

Fully Convolutional Network (https://arxiv.org/abs/1605.06211) was released in 2015 and achieved a performance of 67.2% mean IU on PASCAL VOC 2012. FCNs uses ILSVRC classifiers which were casted into fully-convolutional networks and augmented for dense prediction using pixel-wise loss and in-network up-sampling. This means that previous classfier like VGG16, AlexNet or GoogleNet are used as Encoders and the Fully Connected Layers or Dense Layers are replaced by convolutions. Then, the downsampled features are upsampled by transpose convolutions. The upsample layers are initialized with bilinear interpolation but then the upsample operation is learned by the network. Skip connection are used to combine fine layers and coarse layers, which lets the model to make local predictions that respect global structure.

You can find the original implementation at https://github.com/shelhamer/fcn.berkeleyvision.org, but it uses Caffe.

### Our Implementation
Our implementation uses a VGG16 network as Encoder. The main differences with the author's implementation are:
* We use of BatchNormalization by default (but it can be disabled)
* We don't use double learning rate for biases
* We use l2 regularization instead of weight decay
* We have support for depthwise separable convolutions

## Depthwise Separable FCN8 (FastFCN8)
Inspired by the recent success of Depthwise Separable Convolution we built the FastFCN8 model. Please, this FastFCN8 model IS NOT the model refered in this [paper](https://paperswithcode.com/paper/fastfcn-rethinking-dilated-convolution-in-the)
This FCN8 implementation include support for Depthwise Separable Convolution which allows the model to run faster and reduce drastically the memory usage from 124M to 20M without losing performance accuracy. The model's performance is tested in the example notebooks provided, click [here]() for more details.

To use standard FCN8 or FastFCN8 build the model changing the conv_type parameter from 'conv' to 'ds' respectively.

```
FCN8 = build_model(*params, conv_type='conv')
FastFCN8 = build_model(*params, conv_type='ds')

```

## U-Net
![unet](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-1.46.43-PM.png)

[Convolutional Networks for Biomedical Image Segmentation (U-Net)](https://arxiv.org/abs/1505.04597) was introduced in 2015.
This paper builds upon the fully convolutional layer and modifies it to work on a few training images and yield more precise segmentation. Ronneberger et al. improve upon the "fully convolutional" architecture primarily through expanding the capacity of the decoder module of the network. More concretely, they propose the U-Net architecture which "consists of a contracting path to capture context and a symmetric expanding path that enables precise localization." This simpler architecture has grown to be very popular and has been adapted for a variety of segmentation problems. ([credit for this information and images](https://www.jeremyjordan.me/semantic-segmentation/))

### MobileNet-Unet
We provide the MobileNet-V2 U-Net version, where a MobileNet-V2 network is used as Encoder and the [Pix2Pix](https://www.tensorflow.org/tutorials/generative/pix2pix) Upsample is used as Decoder. This model was obtained and adapted from Tensorflow Documentation, for more information follow [this](https://www.tensorflow.org/tutorials/images/segmentation) link.

To build mobilenet-unet model
