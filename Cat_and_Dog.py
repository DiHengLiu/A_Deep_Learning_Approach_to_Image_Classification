
# coding: utf-8

# In[ ]:


import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# In[ ]:


train = '/Users/dihengliu/Desktop/kagglecatsanddogs_3367a/PetImages'
valid = '/Users/dihengliu/Desktop/kagglecatsanddogs_3367a/valid'

train_batches = ImageDataGenerator().flow_from_directory(train, target_size = (224,224), 
                                                         classes = ['Cat','Dog'], 
                                                         batch_size = 20)
valid_batches = ImageDataGenerator().flow_from_directory(valid, target_size = (224,224), 
                                                         classes = ['Cat','Dog'], 
                                                         batch_size = 2)


# In[ ]:


imgs, lables = next(train_batches)
print(imgs[0], lables[0])
print(imgs[0].shape)


# In[ ]:


imgs.shape


# from keras.models import Sequential
# from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
# num_classes = 2
# model = Sequential()
# model.add(Conv2D(32, (3,3), padding = 'same', input_shape = (256, 256, 3)))
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3,3), padding = 'same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(64, (3,3), padding = 'same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3,3), padding = 'same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size = (2,2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('softmax'))
# model.summary()

# In[ ]:


import numpy as np
x = np.array([j[0] for i in imgs[0] for j in i])
print(x)
plt.imshow(x.reshape(224,224)) #, cmap = 'Greys')


# input_shape = imgs[0].shape
# num_class = 2
# 
# model = Sequential()
# model.add(Conv2D(32, 
#                  kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape = input_shape))
# 
# model.add(Conv2D(32, 
#                  kernel_size = (3, 3), 
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
# 
# model.add(Conv2D(64, 
#                  kernel_size=(3, 3),
#                  activation='relu'))
# 
# model.add(Conv2D(64, 
#                  kernel_size = (3, 3), 
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Conv2D(64, 
#                  kernel_size=(3, 3),
#                  activation='relu'))
# 
# model.add(Conv2D(64, 
#                  kernel_size = (3, 3), 
#                  activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# 
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_class, activation='softmax'))
# 
# model.summary()

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit_generator(train_batches, 
#                     steps_per_epoch = 4, 
#                     validation_data = valid_batches, 
#                     validation_steps = 4,
#                     epochs = 2, 
#                     verbose = 1)

# In[ ]:


vgg16_model = keras.applications.vgg16.VGG16()

vgg16_model.summary()
type(vgg16_model)


# In[ ]:


model1 = Sequential()
for layer in vgg16_model.layers[:-1]:
    model1.add(layer)

model1.summary()
type(model1)


# In[ ]:


for layer in model1.layers:
    layer.trainable = False

model1.add(Dense(2,activation = 'softmax'))
model1.summary()


# In[ ]:


model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.adam(),
              metrics=['accuracy'])


# In[ ]:


model1.fit_generator(train_batches, 
                    steps_per_epoch = 20, 
                    validation_data = valid_batches, 
                    validation_steps = 4,
                    epochs = 2, 
                    verbose = 1)

