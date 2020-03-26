import tensorflow as tf 
import numpy as np 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization,\
						UpSampling2D,AveragePooling2D,Lambda, add, Concatenate,\
						DepthwiseConv2D, Dropout, Softmax, ReLU, Add, MaxPool2D, Conv2DTranspose,\
						Cropping2D
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Zeros, TruncatedNormal
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import sparse_categorical_accuracy as sparse_acc
from math import ceil

def conv_block(inputs,conv_type, kernel, kernel_size, strides,num_conv_layers=3,\
				name=None, padding='same',use_pooling=True, use_relu=True, use_bn=True, l2_reg=0.):
	"""
	Define the basic convoutional blocks of FCN8
	Parameters:
		inputs: Input tensor
		conv_type: 'ds' for depthwise separable convolutions, in other case standard convolution operation
		kernel: Number of filters (kernels)
		kernel_size: A 2D value (filter_height,filter_width)
		strides: A 2D value (stride_vertical,stride_horizontal)
		conv_layers: Number of convolutions to be used in the block. By definition of FCN8, value in [2,3]
		name: Name of the layer
		padding: ('same','valid')
		use_relu: Wheather or not use ReLU as activation function
		use_bn: Wheather or not use batch normalization
		use_pooling: Wheather or not use MaxPooling
		l2_reg: l2 regularizer value
	"""
	last_layer = inputs
	for i in range(1,num_conv_layers+1):
		nname = name+'_'+str(i)
		if(conv_type == 'ds'):
			x = SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides, name=nname,\
				 pointwise_regularizer=l2(l2_reg),depthwise_regularizer=l2(l2_reg))(last_layer)
		else:
			x = Conv2D(kernel, kernel_size, padding=padding, strides = strides, name=nname, kernel_regularizer=l2(l2_reg))(last_layer)  
		if use_bn:
			x = BatchNormalization(name=nname+'_bn')(x)
		if use_relu:
			x = ReLU(name=nname+'_relu')(x)
		last_layer = x
	if use_pooling:
		x = MaxPool2D(pool_size=(2,2),strides=(2,2),name=name+'_mpool',padding='same')(x)
	return x

def fc_block(inputs,conv_type,kernel, kernel_size, strides,dropout, name=None, padding='same', use_relu=True, use_bn=False, l2_reg=0., use_dropout=True):
	"""
	FCN8 replace fully connected layers to use a fully convolutional model
	Parameters:
		inputs: Input tensor
		conv_type: 'ds' for depthwise separable convolutions, in other case standard convolution operation
		kernel: Number of filters (kernels)
		kernel_size: A 2D value (filter_height,filter_width)
		strides: A 2D value (stride_vertical,stride_horizontal)
		dropout: Float between 0 and 1. Fraction of the input units to drop.
		name: Name of the layer
		padding: ('same','valid')
		use_relu: Wheather or not use ReLU as activation function
		use_bn: Wheather or not use batch normalization
		l2_reg: l2 regularizer value
	"""
	if conv_type == 'ds':
		x = SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides, name=name,\
				 pointwise_regularizer=l2(l2_reg),depthwise_regularizer=l2(l2_reg))(inputs)
	else:
		x = Conv2D(kernel, kernel_size, padding=padding, strides = strides, name=name, kernel_regularizer=l2(l2_reg))(inputs)
	if use_dropout:
		x = Dropout(rate=1.-dropout, name=name+'_dropout')(x)
	x = ReLU(name=name+'_relu')(x)
	return x

def score_block(inputs,conv_type,num_classes,name,l2_reg=0.):
	"""
	1x1 Convolution applied in FCN8 to compute class scores at different levels of the network.
	Parameters:
		inputs: Input tensor
		conv_type: 'ds' for depthwise separable convolutions, in other case standard convolution operation
		num_classes: Number of classes to classify
		name: Name of the layer. 'score32' or 'score16' or 'score8'
		l2_reg: l2 regularizer value
	"""
	#He initialization
	if name=='score32':
		in_channels = inputs.shape[-1]
		stddev = (2/in_channels)**.5
	elif name=='score16':
		stddev = 0.01
	elif name=='score8':
		stddev = .001
	w_initializer = TruncatedNormal(stddev=stddev)
	b_initializer = Zeros()
	if conv_type=='ds':
		x = SeparableConv2D(num_classes, 1, padding='same', strides=1, name=name, depthwise_initializer=w_initializer,\
			pointwise_initializer=w_initializer, bias_initializer=b_initializer,\
			pointwise_regularizer=l2(l2_reg), depthwise_regularizer=l2(l2_reg))(inputs)
	else:
		x = Conv2D(num_classes, (1,1), padding='same', strides = (1,1), name=name,\
						kernel_initializer=w_initializer, bias_initializer=b_initializer, kernel_regularizer=l2(l2_reg))(inputs)
	return x

def crop(target_layer,name):
	def wrapper(input):
		h,w = input.get_shape()[1],input.get_shape()[2]
		target_h, target_w = target_layer.get_shape()[1],target_layer.get_shape()[2]
		off_h = h-target_h
		off_w = w-target_w
		offsets = ((off_h//2,ceil(off_h/2)),(off_w//2,ceil(off_w/2)))
		x = Cropping2D(offsets,name=name)(input)
		return x
	return wrapper

def upsample_block(inputs,num_classes, ksize=4,stride=2, name=None, l2_reg=0,use_bias=False):
	"""
	Upsample the input tensor
	Paramters:
		inputs: Input tensor to be upsampled
		num_classes: Number of classes to be clasiffied
		ksize: Kernel size to be used. (uint8)
		stride: Stride value to be used (uint8)
		name: Layer's name
		l2_reg: l2 regularizer value
		use_bias: Wheather or not use biases
	"""
	x = Conv2DTranspose(num_classes,(ksize, ksize),strides=(stride,stride),padding='same',\
						name=name,use_bias=False,kernel_regularizer=l2(l2_reg))(inputs)
	return x


def build_model(w,h,num_classes, dropout=.5, l2_reg=0.,conv_type='ds'):
	X = Input(shape=(h,w,3),name='X')
	#conv1 = conv_block(X,conv_type,64,3,1, name='conv1',num_conv_layers=2,l2_reg=l2_reg)
	#conv2 = conv_block(conv1,conv_type,128,3,1,name='conv2',l2_reg=l2_reg)
	conv1 = conv_block(X,conv_type,64,3,1, name='conv1',num_conv_layers=2,l2_reg=l2_reg,use_bn=True)
	conv2 = conv_block(conv1,conv_type,128,3,1,name='conv2',num_conv_layers=2,l2_reg=l2_reg,use_bn=True)
	conv3 = conv_block(conv2,conv_type,256,3,1,name='conv3',l2_reg=l2_reg,use_bn=True)
	conv4 = conv_block(conv3,conv_type,512,3,1,name='conv4',l2_reg=l2_reg,use_bn=True)
	conv5 = conv_block(conv4,conv_type,512,3,1,name='conv5',l2_reg=l2_reg,use_bn=True)

	fc1 = fc_block(conv5,conv_type,4096,(7,7), strides=(1,1), dropout=dropout, name='fc1',l2_reg=l2_reg,use_dropout=True,use_bn=True)
	fc2 = fc_block(fc1,conv_type,4096,(1,1), strides=(1,1), dropout=dropout, name='fc2',l2_reg=l2_reg,use_dropout=True,use_bn=True)

	score32 = score_block(fc2,conv_type,num_classes,name='score32',l2_reg=l2_reg)
	score16 = score_block(conv4,conv_type,num_classes,name='score16',l2_reg=l2_reg)
	score8 = score_block(conv3,conv_type,num_classes,name='score8',l2_reg=l2_reg)

	upscore32 = upsample_block(score32,num_classes,4,2,name='upscore32',l2_reg=l2_reg)
	upscore32c = crop(score16,name='upscore32c')(upscore32)
	fuse1 = Add(name='fuse1')([score16, upscore32c])

	upscore16 = upsample_block(fuse1,num_classes,4,2,name='upscore16',l2_reg=l2_reg)
	upscore16c = crop(score8,name='upscore16c')(upscore16)
	fuse2 = Add(name='fuse2')([upscore16c, score8])

	upscore8 = UpSampling2D((8,8),name='upscore8')(fuse2)
	upscore8c = crop(X,name='upscore8c')(upscore8)
	classifier = Lambda(lambda x: softmax(x))(upscore8c)

	fcn8 = Model(inputs=X, outputs=classifier, name = 'FCN8')
	return fcn8