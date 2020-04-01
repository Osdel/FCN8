"""
Helper Functions for FC_Densenet Model
"""
def transition_up(skip,block,stride=2):
  """
  Upsample the input block and then concatenate it with the skip.
  The dimension of the skip layer must be 2x the dimension of the block to be able to concatenate them
  Parameters:
    skip: Skip layer to concatenate. 
    block: Target block to upsample.
  """
  filters_to_keep = K.int_shape(skip)[-1]//2
  x = Conv2DTranspose(filters_to_keep,3,stride,padding='same',kernel_initializer='he_uniform')(block)
  x = Concatenate()([x,skip])
  return x

def score(inputs, n_classes, mode='softmax'):
    """
    Performs 1x1 convolution followed by nonlinearity
    Parameters:
      inputs: Input layer
      n_classes: Number of classes
      mode: The nonlinearity function to be used. softmax by default. 
    """
    x = Conv2D(n_classes, kernel_size=1, padding='same', kernel_initializer='he_uniform')(inputs)
    x = Activation(mode)(x)
    return x

"""
The following functions were obtained from [keras repository]
(https://github.com/keras-team/keras-applications/blob/bb8618db8d764e85159b898688c269312fa386b/keras_applications/densenet.py#L93)
"""

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x