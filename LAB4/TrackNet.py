from keras.models import *
from keras.layers import *
from keras.activations import *

'''
1. Skip connection = Upsampling + Concatenate
https://www.tensorflow.org/api_docs/python/tf/keras/layers/UpSampling2D
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Concatenate

Upsampling Ex: 
>>> input_shape = (2, 2, 1, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> print(x.shape)
>>> print(x)
(2, 2, 1, 3)
[[[[ 0  1  2]]
 [[ 3  4  5]]]
 [[[ 6  7  8]]
  [[ 9 10 11]]]]

>>> x2 = tf.keras.layers.UpSampling2D(size=(2, 2),data_format='channels_first')(x)
>>>> print(x2)
tf.Tensor(
[[[[ 0  0  1  1  2  2]
   [ 0  0  1  1  2  2]]
  [[ 3  3  4  4  5  5]
   [ 3  3  4  4  5  5]]]
 [[[ 6  6  7  7  8  8]
   [ 6  6  7  7  8  8]]
  [[ 9  9 10 10 11 11]
   [ 9  9 10 10 11 11]]]], shape=(2, 2, 2, 6), dtype=int64)

Concatenate Ex:
>>> x = tf.convert_to_tensor(x, dtype=tf.float32)
>>> x3 = concatenate( [x, x], axis=1)
>>> print(x3.shape)
>>> print(x3.numpy())
(2, 4, 1, 3)
[[[[ 0.  1.  2.]]
  [[ 3.  4.  5.]]
  [[ 0.  1.  2.]]
  [[ 3.  4.  5.]]]

 [[[ 6.  7.  8.]]
  [[ 9. 10. 11.]]
  [[ 6.  7.  8.]]
  [[ 9. 10. 11.]]]]

the layer you want to imply short connection(32*32*256),you shold upsampling(64*64*256),and concatenate with others(64*64*256)  --> (128*128*256) 
'''


def TrackNet( input_height, input_width ): #input_height = 288, input_width = 512

	imgs_input = Input(shape=(3,input_height,input_width)) #input channel --> data_format='channels_first' 

	#Layer1
	x = Conv2D(32, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(imgs_input)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer2
	x = Conv2D(32, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	drop0 = Dropout(0.5)(x)
	x = drop0

	#Layer3
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#Layer4
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer5
	x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	drop1 = Dropout(0.5)(x)
	x = drop1
	
	#Layer6
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#Layer7
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer8
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer9
	x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	drop2 = Dropout(0.5)(x)
	x = drop2

	#Layer10
	x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first' )(x)

	#Layer11
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer12
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer13
	x = ( Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer14
	x = UpSampling2D((2,2), data_format='channels_first')(x)

	#Layer15
	x = ( Conv2D(128, (2, 2), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	x = concatenate([drop2,x], axis = 1)

	#Layer16
	x = ( Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer17
	x = ( Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first'))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)
	
	#Layer18
	x = UpSampling2D((2,2), data_format='channels_first')(x)

	#Layer19
	x = ( Conv2D(64 , (2, 2), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	x = concatenate([drop1,x], axis = 1)

	#Layer20
	x = ( Conv2D(64 , (3, 3), kernel_initializer='random_uniform', padding='same' , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer21
	x = UpSampling2D( (2,2), data_format='channels_first')(x)

	#Layer22
	x = ( Conv2D(32 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	x = concatenate([drop0,x], axis = 1)

	#Layer23
	x = ( Conv2D(32 , (3, 3), kernel_initializer='random_uniform', padding='same'  , data_format='channels_first' ))(x)
	x = ( Activation('relu'))(x)
	x = ( BatchNormalization())(x)

	#Layer24
	x =  Conv2D( 1 , (1, 1) , kernel_initializer='random_uniform', padding='same', data_format='channels_first' )(x)
	x = ( Activation('sigmoid'))(x)
        

	o_shape = Model(imgs_input , x ).output_shape

	#print ("layer24 output shape:", o_shape[1],o_shape[2],o_shape[3])
	#Layer24 output shape: (1, 288, 512)

	OutputHeight = o_shape[2]
	OutputWidth = o_shape[3]

	output = x

	model = Model( imgs_input , output)
	#model input unit:3*288*512, output unit:1*288*512
	model.outputWidth = OutputWidth
	model.outputHeight = OutputHeight

	#Show model's details
	#model.summary()

	return model




