#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier as mlp 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv("dataset_44_spambase.csv")


# In[3]:


data.head()


# In[4]:


print(data.shape)


# In[5]:


X1=np.array(data)
X=X1[:,0:57]
y=X1[:,57]


# In[6]:



print(X.shape)
#print(data.shape)
#data.head()
#y.reshape(4601,1)
print(y)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,test_size=0.20, random_state=101)


# In[8]:


clf=LogisticRegression(solver='liblinear').fit(X_train,y_train)


# In[9]:


print(clf.score(X_train,y_train))
pred=clf.predict(X_test)
y1=np.ones(y_test.shape)
#print(y1)
print((clf.score(X_test,y_test)))

#print((clf.score(X_test,y1)))


# In[10]:


clf2=mlp(hidden_layer_sizes=(1000,1000,1000),activation='tanh',solver='adam',max_iter=2000)
clf2.fit(X_train,y_train)
print(clf2.score(X_train,y_train))
print(clf2.score(X_test,y_test))


# In[ ]:





# In[11]:


clf3=mlp(hidden_layer_sizes=(64,32,64),activation='tanh',solver='adam',max_iter=500)
clf3.fit(X_train,y_train)
print(clf3.score(X_train,y_train))
print(clf3.score(X_test,y_test))


# In[165]:


# print(data.__doc__)

