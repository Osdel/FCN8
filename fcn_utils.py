import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization,AveragePooling2D,\
					Lambda,DepthwiseConv2D, Dropout, ReLU, Add, MaxPool2D, Conv2DTranspose,Cropping2D
from tensorflow.keras.activations import softmax
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.regularizers import l2
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
		in_channels = inputs.get_shape().as_list()[-1]
		stddev = (2/in_channels)**.5
	elif name=='score16':
		stddev = 0.01
	elif name=='score8':
		stddev = .001
	#w_initializer = TruncatedNormal(stddev=stddev)
	w_initializer = tf.keras.initializers.he_normal()
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
		input_shape = input.get_shape().as_list()
		target_shape = target_layer.get_shape().as_list()
		h,w = input_shape[1],input_shape[2]
		target_h, target_w = target_shape[1],target_shape[2]
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