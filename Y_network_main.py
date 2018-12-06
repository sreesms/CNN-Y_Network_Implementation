import numpy as np 
from keras.layers import Dense,Dropout, Input
from keras.layers import Conv2D, MaxPooling2D,Flatten
from keras.models import Model 
from keras.layers.merge import concatenate
from keras.datasets import mnist 
from keras.utils import to_categorical
from keras.utils import plot_model

#load mnist dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#compute numbr of labels
num_labels = len(np.unique(y_train))

#convert to one code vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#reshape and normalize
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1,image_size,image_size,1])
x_test = np.reshape(x_test,[-1,image_size,image_size,1])
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#network parameters
input_shape = (image_size,image_size,1)
batch_size = 50
kernel_size = 3
dropout = 0.4
n_flilters = 16

left_inputs = Input(shape=input_shape)
x= left_inputs
filters = n_flilters

#4 layers of Conv2D- Dropout - maxpooling2D
for i in range(4):
	x = Conv2D(filters=filters,
		kernel_size= kernel_size,
		padding='same',
		activation = 'relu')(x)
	x= Dropout(dropout)(x)
	x= MaxPooling2D()(x)
	filters*=2

#right_inputs 
right_inputs = Input(shape = input_shape)
y = right_inputs

#4 layers 
for i in range(4):
	y = Conv2D(filters=filters,
			kernel_size = kernel_size,
			padding = 'same',
			activation = 'relu',
			dilation_rate=3)(y)
	y = Dropout(dropout)(y)
	y= MaxPooling2D()(y)
	filters *=2

#Merge left and right branches outputs
y = concatenate([x,y])

#feature maps to vector before connecting to Dense layer
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(num_labels,activation='softmax')(y)

#building model
model = Model([left_inputs,right_inputs],outputs)

#verify the model
# plot_model(model,to_file = 'cnn-y-network.png',show_shapes = True)
model.summary()

#classifier loss, Adam Optimizer, classifier accuracy
model.compile(loss = 'categorical_crossentropy',
				optimizer = 'adam',
				metrics =['accuracy'])

#train the model with input images and labels
model.fit([x_train,x_train],y_train,validation_data = ([x_test,x_test],y_test),epochs = 30,batch_size= batch_size)

#accuracy on test data
score = model.evaluate([x_test,x_test],y_test,batch_size=batch_size)

print("\nTest Accuracy %.lf%%" %(100.0*score[1]))
