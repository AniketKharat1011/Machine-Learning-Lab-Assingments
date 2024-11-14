#!/usr/bin/env python
# coding: utf-8

# # Name = Aniket Kharat
# # Roll no : 2447008 

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[2]:


df= pd.read_csv('Iris.csv')


# In[3]:


df


# In[4]:


df.drop('Id',axis=1,inplace=True)


# In[5]:


df


# In[6]:


X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]


# In[7]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[8]:


wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# In[9]:


plt.figure(figsize=(8,5))
plt.plot(range(1,11),wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS (Within-cluster sum of squares)')
plt.grid(True)
plt.show()


# In[11]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10,random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[12]:


df['Cluster'] = y_kmeans


# In[ ]:


plt.figure(figsize=(8,5))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0,2], s=100, c='red')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1,2], s=100, c='blue')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2,2], s=100, c='red')

