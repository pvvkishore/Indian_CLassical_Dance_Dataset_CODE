#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:43:39 2023

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 09:55:24 2022

@author: root
"""

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from matplotlib.pyplot import imshow
import scipy.misc
from keras.initializers import glorot_uniform
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.models import Model, load_model
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import PIL
batch_size = 32
img_height = 224
img_width = 224
dataset_path = os.listdir('Cropped Online Dataset')
dance_types = os.listdir('Cropped Online Dataset')
dance_types = sorted(dance_types)
print(dance_types)
#%%
""" Store all the images in a  single list """
import cv2
Dance_list = []
image = []
labels = []
for item in dance_types:
    all_images_class = os.listdir('Cropped Online Dataset'+'/'+item)
    for images in all_images_class:
        img = cv2.imread(str('Cropped Online Dataset'+'/'+item+'/'+images))
        img = cv2.resize(img,[224,224])
        image.append(img)
        labels.append(item)
        Dance_list.append((item,str('Cropped Online Dataset'+'/'+item+'/'+images)))
print(len(Dance_list))
imagea = np.array(image)
imagea = imagea.astype('float32')/255
print(imagea.shape)
#%%
#Convert the dataset to a list for random selection
def dataset_seletor(train_data):
    train_data_list = list(train_dataset.as_numpy_iterator())

    # Set the number of images you want to select
    num_images_to_select = 3

    # Randomly select images and labels
    selected_frames = []
    selected_labels = []

    if len(train_data_list) >= num_images_to_select:
        selected_indices = np.random.choice(len(train_data_list), num_images_to_select, replace=False)

        for idx in selected_indices:
            image, label = train_data_list[idx]
            selected_images.append(image)
            selected_labels.append(label)

    # Convert the selected images and labels back to TensorFlow tensors if needed
    X_train = tf.convert_to_tensor(selected_images)
    y_train = tf.convert_to_tensor(selected_labels)
    return X_train, y_train
#%%
"""Endode the labels"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y_labelencoder = LabelEncoder()
Y = y_labelencoder.fit_transform(labels)
Y2 = Y.reshape(-1,1)
enc = OneHotEncoder(handle_unknown='ignore')
Y1 = enc.fit_transform(Y2)
y = Y1.toarray()
print(y.shape)
#%%
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.wq = [Dense(self.depth) for _ in range(num_heads)]
        self.wk = [Dense(self.depth) for _ in range(num_heads)]
        self.wv = [Dense(self.depth) for _ in range(num_heads)]
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq[i](q) for i in range(self.num_heads), batch_size)
        k = self.split_heads(self.wk[i](k) for i in range(self.num_heads), batch_size)
        v = self.split_heads(self.wv[i](v) for i in range(self.num_heads), batch_size)

        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, d_model))
        return self.dense(output)
#%%
def identity_block(x, f, filters):
    F1, F2, F3 = filters
    x_shortcut = x

    # First Layer
    x = Conv2D(filters=F1, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Second Layer
    x = Conv2D(filters=F2, kernel_size=(f, f),
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # Third layer
    x = Conv2D(filters=F3, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Add()([x, x_shortcut])
    x = Activation('relu')(x)

    return x

#%%
def convolutional_block(x, f, filters, s=2):
    F1, F2, F3 = filters
    x_convcut = x
    # First Layer
    x = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Second Layer
    x = Conv2D(filters=F2, kernel_size=(f, f),
               strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Third layer
    x = Conv2D(filters=F3, kernel_size=(1, 1),
               strides=(1, 1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    # Short cut path with conv
    x_convcut = Conv2D(filters=F3, kernel_size=(
        1, 1), strides=(s, s), padding='valid')(x_convcut)
    x_convcut = BatchNormalization(axis=3)(x_convcut)

    x = Add()([x, x_convcut])
    x = Activation('relu')(x)

    return x

#%%
def MFMHAResNet50(input_shape=(224, 224, 3), classes=5, num_of_heads):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)

    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2 Convolutional block
    x = convolutional_block(x, f=3, filters=[64, 64, 256], s=1)

    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    attention = MultiHeadAttention(d_model=256, num_heads=num_of_heads)(x, x, x, mask=None)
    # Stage 3
    x = convolutional_block(attention, f=3, filters=[128, 128, 512], s=2)
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    attention = MultiHeadAttention(d_model=256, num_heads=num_of_heads)(x, x, x, mask=None)
    # Stage 4
    x = convolutional_block(attention, f=3, filters=[256, 256, 1024], s=2)
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    attention = MultiHeadAttention(d_model=256, num_heads=num_of_heads)(x, x, x, mask=None)
    # Stage 5
    x = convolutional_block(attention, f=3, filters=[512, 512, 2048], s=2)
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    attention = MultiHeadAttention(d_model=256, num_heads=num_of_heads)(x, x, x, mask=None)
    x = AveragePooling2D((2, 2), name='avg_pool2D')(attention)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc'+str(classes),
              kernel_initializer=glorot_uniform(seed=0))(x)

    # Create Model
    model = Model(inputs=x_input, outputs=x, name='ResNet50')

    return model
#%%
#%%
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystopper = EarlyStopping(monitor = 'val_loss', patience = 10,
                              verbose = 1, restore_best_weights = True)
reduce1 = ReduceLROnPlateau(monitor = 'val_loss', patience = 10,
                            verbose = 1, factor = 0.5, min_lr = 1e-6)
optimizer = Adam(1e-7)
model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

history = MFMHAResNet50.fit([X1_train,X2_train,X3_train], 
                    y_train, epochs = 200, 
                    batch_size = 32, verbose=1,
                    validation_data=([X1_test,X2_test, X3_test],
                                     y_test))
#%% Plot the attention maps
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
#%% Load original image from disk (in openCV format) and resize it
img_path = '/home/user/Desktop/Classical Dance Identification_DL/Cropped Online Dataset/1/Nikolina Nikoleski_ Bho Shambo 01836.jpg'
orig = cv2.imread(img_path)  
resized = cv2.resize(orig, (224,224))
# Load image in tf format       
image = load_img(img_path, target_size=(224,224))
image = img_to_array(image)
image /= 255
image = np.expand_dims(image, axis = 0)
image = imagenet_utils.preprocess_input(image)

# cv2.imshow('img',orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 
#%%# Use network to make predictions on the input image and find
# the class label index with the largest probability
preds = model.predict([image,image1,image2,image3])      
i = np.argmax(preds[0])
print(i)
#%%
#import seaborn as sns
import matplotlib.pyplot as plt
train_data = {'loss': history.history['loss'],
              'val_loss': history.history['val_loss'],
              'acc': history.history['accuracy'],
              'val_acc': history.history['val_accuracy']}
#sns.set_style('whitegrid')
fig, (ax1,ax2) = plt.subplots(2, 1, sharex = 'col',
                             figsize = (20, 14))
ax1.plot(history.history['loss'], label = 'Train Loss', linewidth = 4)
ax1.plot(history.history['val_loss'], label = 'validation Loss', linewidth = 4)
ax1.legend(loc = 'best')
ax1.set_title('Loss')
leg = ax1.legend()
leg_lines = leg.get_lines()
leg_text = leg.get_texts()
plt.setp(leg_lines, linewidth = 8)
plt.setp(leg_text, fontsize = 'x-large')
#plt.show()

ax2.plot(history.history['accuracy'], label='Train Accuracy', linewidth = 4)
ax2.plot(history.history['val_accuracy'], label='Validation accuracy', linewidth = 4)
ax2.legend(loc='best')
ax2.set_title('Accuracy')
plt.xlabel('Epochs')

leg = ax2.legend()
leg_lines = leg.get_lines()
leg_text = leg.get_texts()
plt.setp(leg_lines, linewidth = 8)
plt.setp(leg_text, fontsize = 'x-large')
plt.show()
#%%
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow
layers_outputs = [layer.output for layer in model.layers[1:]]
#print(layers_outputs)
visualize_model = tensorflow.keras.models.Model(inputs = model.input, 
                                                outputs = layers_outputs)
feature_maps0 = visualize_model.predict([image,image1,image2,image3])
print(len(feature_maps0))
layer_names = [layer.name for layer in model.layers]
print(layer_names)
#%%
# import matplotlib.pyplot as plt
# #get_ipython().run_line_magic('matplotlib', 'inline')
# import cv2
# file = "D:\Dance_Project_2022\results"
# import glob
# for layer_names, feature_maps0 in zip(layer_names, feature_maps0):
#     print(feature_maps0.shape)
#     if len(feature_maps.shape) == 4:
#         channels = feature_maps.shape[-1]
#         size = feature_maps.shape[1]
#         display_grid = np.zeros((size,size*channels))
#         for i in range(channels):
#             x = feature_maps[1][0,:,:,i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 224
#             x += 224
#             x = np.clip(x,0,255).astype('uint8')
#             # tile each filter into a big horizontal grid
#             display_grid[:,i*size:(i+1)*size] = x
            
#             scale = 2./channels
#             plt.figure(figsize=(scale*channels, scale))
#             plt.title(layer_names)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
#         cv2.imwrite('file\messigray{}.png'.format(layer_names), x)
# #%%
# import matplotlib.pyplot as plt
# #get_ipython().run_line_magic('matplotlib', 'inline')
# import cv2
# file = "/home/user/Desktop/Classical Dance Identification_DL\results"
# import glob
# feature_maps = feature_maps0[5] 
# feature_maps = np.array(feature_maps)
# if len(feature_maps.shape) == 4:
#         channels = feature_maps.shape[-1]
#         size = feature_maps.shape[1]
#         display_grid = np.zeros((size,size*channels))
#         for i in range(channels):
#             x = feature_maps[0,:,:,i]
#             x -= x.mean()
#             x /= x.std()
#             x *= 224
#             x += 224
#             x = np.clip(x,0,255).astype('uint8')
#             y = np.clip(x,0,255).astype('uint8')
#             x1 = y
#             # tile each filter into a big horizontal grid
#             display_grid[:,i*size:(i+1)*size] = x
            
#             scale = 2./channels
#             plt.figure(figsize=(scale*channels, scale))
#             plt.title(layer_names)
#             plt.grid(False)
#             plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')
#         cv2.imwrite('file\messigray{}.png'.format(layer_names[5]), x1)
#%%
for layer_names, feature_maps0 in zip(layer_names, feature_maps0):
    savedir = {'layer_names': layer_names,
               'feature_maps': feature_maps0}
#%%
import scipy.io as sio
sio.savemat('my_arrays.mat', savedir)
