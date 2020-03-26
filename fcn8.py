import tensorflow as tf 
import numpy as np 
from tensorflow.keras.layers import Input, UpSampling2D,Lambda,Add
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model
from fcn_utils import *

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