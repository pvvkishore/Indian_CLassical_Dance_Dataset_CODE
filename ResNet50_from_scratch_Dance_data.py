#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow as tf
dataset_path = os.listdir('Cropped Online Dataset')


# In[2]:


dance_types = os.listdir('Cropped Online Dataset')
dance_types


# # Store all the images into a single list

# In[3]:


dance_list = []
for item in dance_types:
    all_dance_type = os.listdir('Cropped Online Dataset'+'/'+item)
    for dance in all_dance_type:
        dance_list.append((item, str('Cropped Online Dataset'+'/'+item)+'/'+dance))


# In[4]:


#dance_list


# # Build a data frame of items in the dataset

# In[5]:


dance_df = pd.DataFrame(data = dance_list, columns = ['pose_class','video_frames'])
dance_df.head()


# # Check the data

# In[6]:


print('Total number of frames in the dataset',len(dance_list))
frame_count = dance_df['pose_class'].value_counts()
print('Frames in each label')
print(frame_count)


# # Read Video Frames from folders

# In[7]:


import cv2
path = 'Cropped Online Dataset/'
im_size = 224
images = []
labels = []
for i in dance_types:
    data_path = path + str(i)
    filenames = [i for i in os.listdir(data_path)]
    for f in filenames:
        img = cv2.imread(data_path+'/'+f)
        img = cv2.resize(img,[im_size,im_size])
        images.append(img)
        labels.append(i)
#images
#labels


# In[8]:


images = np.array(images)
print(images.shape)


# In[9]:


images = images.astype('float32')/255


# In[10]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y = dance_df['pose_class'].values
y_labelencoder = LabelEncoder()
y = y_labelencoder.fit_transform(y)


# In[11]:


print(y)


# In[12]:


from sklearn.compose import ColumnTransformer
y = y.reshape(-1,1)
#onehotencoder = ColumnTransformer(['y',OneHotEncoder(),[0]], remainder = 'passthrough')
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y)
Y = y.toarray()


# In[13]:


print(Y.shape)


# In[14]:


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
images, Y = shuffle(images, Y, random_state = 1)
train_x, test_x, train_y, test_y = train_test_split(images,Y, test_size = 0.3, random_state = 2)


# In[15]:


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# # ResNet50 Design

# In[16]:


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense,Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model

from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


# In[17]:


def identity_block(x,f,filters):
    F1, F2, F3 = filters
    x_shortcut = x
    
    # First Layer
    x = Conv2D(filters = F1, kernel_size = (1,1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Second Layer
    x = Conv2D(filters = F2, kernel_size = (f,f), strides=(1,1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # Third layer
    x = Conv2D(filters = F3, kernel_size = (1,1), strides=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)
    
    return x    


# In[18]:


def convolutional_block(x, f, filters, s=2):
    F1,F2,F3 = filters
    x_convcut = x
    # First Layer
    x = Conv2D(filters = F1,kernel_size = (1,1), strides = (s,s))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Second Layer
    x = Conv2D(filters = F2,kernel_size = (f,f), strides = (1,1), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Third layer
    x = Conv2D(filters = F3,kernel_size = (1,1), strides = (1,1),padding = 'valid')(x)
    x = BatchNormalization(axis=3)(x)
    
    # Short cut path with conv
    x_convcut = Conv2D(filters= F3,kernel_size = (1,1), strides = (s,s), padding = 'valid')(x_convcut)
    x_convcut = BatchNormalization(axis = 3)(x_convcut)
    
    x = Add()([x, x_convcut])
    x = Activation('relu')(x)
    
    return x    


# # Build the ResNet Archtecture

# In[19]:


def ResNet50(input_shape = (224,224,3), classes = 8):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3,3))(x_input)
    
    # Stage 1
    x = Conv2D(64,(7,7), strides = (2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3), strides=(2,2))(x)
    
    # Stage 2 Convolutional block
    x = convolutional_block(x, f=3, filters = [64,64,256], s=1)
    
    x = identity_block(x, 3, [64,64,256])
    x = identity_block(x, 3, [64,64,256])
    
    # Stage 3
    x = convolutional_block(x, f=3, filters = [128,128,512], s=2)
    x = identity_block(x, 3, [128,128,512])
    x = identity_block(x, 3, [128,128,512])
    x = identity_block(x, 3, [128,128,512])
    
    # Stage 4
    x = convolutional_block(x, f=3, filters = [256,256,1024], s=2)
    x = identity_block(x, 3, [256,256,1024])
    x = identity_block(x, 3, [256,256,1024])
    x = identity_block(x, 3, [256,256,1024])
    x = identity_block(x, 3, [256,256,1024])
    x = identity_block(x, 3, [256,256,1024])
    
    # Stage 5
    x = convolutional_block(x, f=3, filters = [512,512,2048], s=2)
    x = identity_block(x, 3, [512,512,2048])
    x = identity_block(x, 3, [512,512,2048])
    
    x = AveragePooling2D((2,2),name = 'avg_pool2D')(x)
    
    x = Flatten()(x)
    x = Dense(classes,activation = 'softmax',name = 'fc'+str(classes),
            kernel_initializer = glorot_uniform(seed=0))(x)
    
    # Create Model
    model = Model(inputs = x_input, outputs=x, name = 'ResNet50')
    
    return model


# In[20]:


model = ResNet50(input_shape = (224,224,3), classes = 8)


# In[21]:


model.summary()


# In[22]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # (n+2p-f)/s+1*(n+2p-f)/s+1 --> Layer Output size calculation
# n = image size
# p = pooling
# f = filter size
# s = stride

# In[23]:


history = model.fit(train_x, train_y, epochs = 3, batch_size = 32)


# In[27]:


pred = model.evaluate(test_x, test_y)
print('Loss = '+str(pred[0]))
print('Test Accuracy = '+str(pred[1]))


# In[ ]:
print(tf.keras.utils.plot_model(model, show_shapes = True))




