# -*- coding: utf-8 -*-
"""Convolutionetiquettageactions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lU4SsGvOR_oBehaneE3fvA05z5DNmFKm
"""


import os



import gym
import minerl


minerl.data.download(directory="/your/local/path/", experiment='MineRLNavigateDense-v0')
import logging
logging.basicConfig(level=logging.DEBUG)





data = minerl.data.make(
    'MineRLNavigateDense-v0',
    data_dir="/your/local/path")

"""On importe tous les packages nécessaires (minerl)"""

import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os






  


"""On importe tous les packages hors-minerl"""

data = np.load("/Mael/MineRLNavigateDense-v0/v1_aggravating_artichoke_harpy-1_5264-6403/rendered.npz")
liste = ["action_attack", "action_back", "action_forward", "action_jump", "action_left", "action_place", "action_right", "action_sneak", "action_sprint", ]

targets = []
features = []
target = []

files = glob.glob('MineRLNavigateDense-v0/*.jpg')

#for i in range(np.size(data[liste[j]])):


#for entry in os.scandir("/your/local/path/MineRLNavigateDense-v0"):

 # data = np.load(os.path.abspath(entry) + "/rendered.npz")
  #for i in range(np.size(data[liste[j]])): #on a une vidéo, le for permet de faire une boucle avec toutes les images de la vidéo
   # for j in range(8):   #pour chaque image on analyse l'action étiquettée
    #  print(data[liste[j]][i]) # 1 si l'action est étiquettée, 0 sinon
     # target[j] = data[liste[j]][i]

    
    #print("fin d'une image")
    #for j in range(8):

  #print("fin d'une vidéo")

for entry in os.scandir("/Mael/MineRLNavigateDense-v0"):
  data = np.load(os.path.abspath(entry) + "/rendered.npz")
  for i in range(np.size(data[liste[0]])):
    
    for j in range(8):
      #print(i)
       #print(data[liste[j]][i])
       #print(liste[j])
       target.append(data[liste[j]][i])
    
    import cv2  #tout le paragraphe transforme une vidéo en images 
    vidcap = cv2.VideoCapture(os.path.basename(entry) + "/recording.mp4")
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite("/Mael/frame"+os.path.basename(entry)+"%d.jpg" % count, image)     # save frame as JPEG file    
      features.append(np.array(Image.open("/Mael/frame "+os.path.abspath(entry)+"%d.jpg" % count).resize((75, 75)))) #réduit la taille de ces images   
      success,image = vidcap.read()
      # print('Read a new frame: ', success)
      count += 1

    

    #print("fin d'une image")
    targets.append(target)


features = np.array(features)
targets = np.array(targets)

print("features shape", features.shape)
print("Targets shape", targets.shape)

"""Maintenant, on a un tableau avec en ligne le nombre d'images, et pour chaque colonne (8) un 0 ou un 1, en fonction de si l'action est étiquettée à l'image.

La data est extraite
"""

from random import randint

for a in [randint(0, len(features)) for _ in range(10)]:
    plt.imshow(features[a], cmap="gray")
    plt.show()

"""Permettait de montrer les photos"""

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(features, targets, test_size=0.05, random_state=42)

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_valid.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_valid.shape)

"""C'était jeu d'entraînement et de validation"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Placeholder
x = tf.placeholder(tf.float32, (None, 75, 75, 3), name="x")
y = tf.placeholder(tf.float32, (None, 2), name="y")
dropout = tf.placeholder(tf.float32, (None), name="dropout")

def create_conv(prev, filter_size, nb):
    # First convolution
    conv_W = tf.Variable(tf.truncated_normal(shape=(filter_size, filter_size, int(prev.get_shape()[-1]), nb)))
    conv_b = tf.Variable(tf.zeros(nb))
    conv   = tf.nn.conv2d(prev, conv_W, strides=[1, 1, 1, 1], padding='SAME') + conv_b
    # Activation: relu
    conv = tf.nn.relu(conv)
    # Pooling
    conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return conv

conv = create_conv(x, 8, 32)
conv = create_conv(conv, 5, 64)
conv = create_conv(conv, 5, 128)
conv = create_conv(conv, 5, 256)
conv = create_conv(conv, 5, 215)

flat = flatten(conv)
print(flat, flat.get_shape()[1])

# First fully connected layer
fc1_W = tf.Variable(tf.truncated_normal(shape=(int(flat.get_shape()[1]), 512)))
fc1_b = tf.Variable(tf.zeros(512))
fc1   = tf.matmul(flat, fc1_W) + fc1_b

# Activation.
fc1    = tf.nn.relu(fc1)

#fc1 = tf.nn.dropout(fc1, keep_prob=dropout)

# Last layer: Prediction
fc3_W  = tf.Variable(tf.truncated_normal(shape=(512, 2)))
fc3_b  = tf.Variable(tf.zeros(2))
logits = tf.matmul(fc1, fc3_W) + fc3_b

softmax = tf.nn.softmax(logits)

"""Création du modèle au dessus. En dessous, erreur et optimisation"""

# Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

# Accuracy
predicted_cls = tf.argmax(softmax, axis=1)
correct_prediction = tf.equal(predicted_cls, tf.argmax(y, axis=1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
training_operation = optimizer.minimize(loss_operation)

"""Entraînement du modèle"""

batch_size = 255

sess =  tf.Session()
sess.run(tf.global_variables_initializer())

from scipy import ndimage
from scipy import misc
from random import randint
import random

def augmented_batch(batch):
    """
    """
    n_batch = []
    
    for img in batch:
        if random.uniform(0, 1) > 0.75:
            process_img = Image.fromarray(np.uint8(img.reshape(75, 75, 3))).rotate(randint(-45, 45))
            n_img = np.array(process_img)
            n_batch.append(n_img.reshape(75, 75, 3))
        else:
            n_batch.append(img)
        
    return n_batch

    i = 0
for epoch in range(0, 10000):
    print(">> Epoch: %s" % epoch)
    # Shuffle
    indexs = np.arange(len(X_train))
    np.random.shuffle(indexs)
    X_train = X_train[indexs]
    y_train = y_train[indexs]
    
    for b in range(0, len(X_train), batch_size):
        batch = augmented_batch(X_train[b:b+batch_size])
        #batch = X_train[b:b+batch_size]

        if i % 20 == 0:
            #print(sess.run(predicted_cls, feed_dict={dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))
            print("Accuracy [Train]:", sess.run(accuracy_operation, feed_dict={dropout: 1.0, x: batch, y: y_train[b:b+batch_size]}))            
        sess.run(training_operation, feed_dict={dropout: 0.8, x: batch, y: y_train[b:b+batch_size]})
        i += 1

    if epoch % 2 == 0:
        accs = []
        for b in range(0, len(X_valid), batch_size):
            accs.append(sess.run(accuracy_operation, feed_dict={dropout: 1., x: X_valid[b:b+batch_size], y: y_valid[b:b+batch_size]}))
        print("Accuracy [Validation]", np.mean(accs))