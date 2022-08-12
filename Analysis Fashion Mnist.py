#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


import pandas as pd
import numpy as np


# In[5]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y) = fashion_mnist.load_data()


# In[6]:


from collections import Counter


# In[7]:


len(Counter(train_y))


# In[8]:


print('Training data dimension:', train_x.shape)
print('Training label dimension:', train_y.shape)
print('Training label:', np.unique(train_y))
print('')
print('Test data dimension:', test_x.shape)
print('Test label dimension:', test_y.shape)
print('Test label:', np.unique(test_y))


# In[9]:


data_dict = {
    0:'T-shirt/Top',
    1:'Trouser',
    2:'Pullover',
    3:'Dress',
    4:'Coat',
    5:'Sandal',
    6:'Shirt',
    7:'Sneaker',
    8:'Bag',
    9:'Ankle boot'
}
print(data_dict)


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


plt.imshow(train_x[3])

ax1 = plt.figure(figsize=(20,10))
count = 1
for i in range(10):
    plt.subplot(2,5,count)
    plt.imshow(train_x[i], cmap='gray')
    plt.title('Label-' + data_dict[train_y[i]])
    plt.axis('off')
    count += 1


# In[12]:


print('Before one-hot processing:', train_y[0])

train_label = tf.keras.utils.to_categorical(train_y,num_classes=10)
test_label = tf.keras.utils.to_categorical(test_y,num_classes=10)

print('After pme-hot processing:', train_label[0])


# In[14]:


train_input = tf.data.Dataset.from_tensor_slices((train_x, train_label)).batch(50)
test_input = tf.data.Dataset.from_tensor_slices((test_x, test_label)).batch(50)

print('Length of training input:', len(train_x))
print('Length of training input after setting Batch:', len(train_input))
train_input


# In[15]:


input_data = tf.keras.Input([28,28])
input_data


# In[16]:


dense = tf.keras.layers.Flatten()(input_data)
dense


# In[17]:


dense = tf.keras.layers.Dense(100, activation='relu')(dense)
dense = tf.keras.layers.Dense(100, activation='relu')(dense)
dense = tf.keras.layers.Dense(100, activation='relu')(dense)
dense = tf.keras.layers.Dense(100, activation='relu')(dense)


# In[18]:


output_data=tf.keras.layers.Dense(10, activation='softmax')(dense)


# In[19]:


model = tf.keras.Model(inputs=input_data,outputs=output_data)


# In[21]:


model.compile(optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.categorical_crossentropy,
            metrics=['accuracy'])


# In[22]:


model.summary()


# In[23]:


model.fit(train_input, epochs=10)


# In[24]:


model.evaluate(test_input)


# In[27]:


ax1 = plt.figure(figsize=(20,10))
count = 1
for i in range(10):
    plt.subplot(2,5,count)
    plt.imshow(train_x[i], cmap='gray')
    
    result = model.predict(tf.expand_dims(test_x[i], axis=0))
    
    plt.title('Actual ' + data_dict[test_y[i]] + ' Predicted as-' + data_dict[np.argmax(result)])
    plt.axis('off')
    count += 1


# In[28]:


train_data = tf.expand_dims(train_x,-1)
test_data = tf.expand_dims(test_x,-1)

train_data.shape


# In[29]:


train_input=tf.data.Dataset.from_tensor_slices((train_data,train_label)).batch(50)
test_input=tf.data.Dataset.from_tensor_slices((test_data,test_label)).batch(50)


# In[30]:


input_data=tf.keras.Input([28,28,1])


# In[31]:


conv=tf.keras.layers.Conv2D(30,5,padding='SAME',activation='relu')(input_data)
conv=tf.keras.layers.Conv2D(30,5,padding='SAME',activation='relu')(conv)


# In[32]:


conv=tf.keras.layers.MaxPool2D(strides=[2,2])(conv)
conv=tf.keras.layers.Conv2D(30,5,padding='SAME',activation='relu')(conv)


# In[34]:


dense=tf.keras.layers.Flatten()(conv)
output_data=tf.keras.layers.Dense(10,activation='softmax')(dense)


# In[35]:


model=tf.keras.Model(inputs=input_data,outputs=output_data)
model.compile(optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.categorical_crossentropy,
            metrics=['accuracy'])
model.summary()


# In[39]:


model.fit(train_input, epochs=10)


# In[40]:


model.evaluate(test_input)


# In[46]:


ax1 = plt.figure(figsize=(20,10))
count = 1
for i in range(10):
    plt.subplot(2,5,count)
    plt.imshow(test_x[i], cmap='gray')
    
    result = modelx.predict(tf.expand_dims(test_x[i], axis=0))
    
    plt.title('Actual ' + data_dict[test_y[i]] + ' Predicted as-' + data_dict[np.argmax(result)])
    plt.axis('off')
    count += 1


# In[42]:


import os


# In[44]:


root_path = os.getcwd()
model.save('./model.h5')

model_path = os.path.join(root_path, 'model.h5')
model.save(model_path)


# In[45]:


modelx = tf.keras.models.load_model('./model.h5')
modelx.evaluate(test_input)


# In[ ]:




