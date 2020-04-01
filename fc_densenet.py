"""
Fully Convolutional Densenet
This implementation uses tf.keras.applications DenseNet as backbone and build a network for
semenatic segmentation using the paper [The One Hundred Layers Tiramisu:
Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326.pdf) as a guideline.
The model has some differences with the paper's implementation, mostly because the use a different DenseNet model
as backbone.
This model is compatible with all DenseNet models provided by tf.keras.applications
"""
from tensorflow.keras.applications import DenseNet169, DenseNet121,DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Concatenate, Softmax, Activation, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras import backend as K
from fc_densenet_utils import *

def build_fc_densenet(n_classes,h,w,n_layers=201,use_bottleneck=False,bottleneck_blocks=32):
  """
  Build a Fully Convolutional Densenet model.
  Parameters:
    n_classes: Number of classes to predict
    h: Height of input images
    w: Width of input images
    n_layers: Numbers of Densenet's layers. Values in [121,169,201]. Densenet201 is used by default or if the value is not in the valid set.
    use_bottleneck: Whether or not use a bottleneck block as mentioned in the paper.
    bottleneck_blocks: Number of blocks to use if use_bottleneck parameter is True
  Return:
    A tf.keras Model instance
  """
  if n_layers == 121:
    blocks = [6, 12, 24, 16]
    base_model = DenseNet121(input_shape=[h, w, 3], include_top=False)
  elif n_layers == 169:
    blocks = [6, 12, 32, 32]
    base_model = DenseNet169(input_shape=[h, w, 3], include_top=False)
  else:
    blocks = [6, 12, 48, 32]
    base_model = DenseNet201(input_shape=[h, w, 3], include_top=False)
  
  skips_n = 3
  grown_factor=32

  #Encoder
  skip_names = [str.format('conv{0}_block{1}_concat',i+2,blocks[i]) for i in range(skips_n+1)]
  upsample_factors = [4,2,2,2]
  skip_layers = [base_model.get_layer(name).output for name in skip_names]
  base = Model(inputs=base_model.inputs,outputs=skip_layers)

  
  inputs = Input(shape=[512, 512, 3])
  skips = base(inputs)

  x = skips[-1]
  #bottleneck
  if use_bottleneck:
    x = dense_block(x,bottleneck_blocks,name='bottleneck')

  #Upsample path
  for i in range(1,4):
    print('upsampling',x,skips[-i-1])
    skip = skips[-i-1]
    x = transition_up(skip,x)
    x = dense_block(x,blocks[-i],name='upsample'+str(i))

  #4x upsampling
  x = Conv2DTranspose(64,3,4,padding='same',kernel_initializer='he_uniform')(x)
  x = score(x,n_classes)

  #ending model
  model = Model(inputs=inputs,outputs=x)
  return model

