#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf


# In[ ]:


(X_train, Y_train), (X_test, Y_tes
t)=tf.keras.datasets.cifar10.loa
d data()
print(X_train.shape)


# In[ ]:


accuracy = []
for k in range(1, 120):
print("k:". k)
acc = []
for f in range(5):
# divide data into folds
validation, validationY = X_t
rain[f*10000: (f+1)*10000], Y_train
[f*10000: (f+1)*10000]
trainX = np. concatenate((X tra
in[: f*10000]. X_train[(f+1)*1000
0:]), axis = 0)
trainY = np. concatenate((Y tra
in[: f*10000], Y_train[ (f+1)*1000
0:]), axis = 0)
cm = ce = 0
 # c for correct, m for manhattan, e fom eucl
idean
for i in range(len(validation X))

man_distance = []
euc distance = []
for j in range(len(trainX)):
# manhattan - calculating distance between two images
l1= np.sum(np.absolute(np. subtract(validationX[i], trainx
[j]))
# euclidean- calculation distance between two images
12 = np.sqrt(np. sum((validationX[i]
- trainX[j])**2))
 man_distance.append([l1, t          
rainY[j][0]])
euc_distance. append([12, t           
rainY[j][0]])
if j== 400:
break
man_distance = np. array(man_distance)
man_distance = man_distance[man_distance[:, 0].argsort()]
euc_distance = np. array(euc_distance)
euc distance = euc_distance[euc_distance[:, 0].argsort()]
# for manhattan distance
values = man_distance[:k, 1]
# print(values)
most_frequent_value = np.arg
max(np.bincount (values.astype('int
32'* )))
# print(most_ frequent_ value)
if most_frequent_value == validationY[i]:
cm + = 1
# for euclidean distance
values = euc_distancel[:k, 1]
most_ frequent_ value = np.arg
max(np.bincount(values.astype('int
32')))
if most frequent value == validationY[i]:
ce += 1
aLue

# print(man_distance[0][0],man_distance [-1][0])
# print (euc_distance[0], euc_distance[-1])
if i == 120:
break
acc. append([cm/100, ce/100])
accuracy.append([acc, k])
print(accuracy)                              




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




