#!/usr/bin/env python
# coding: utf-8

# # Name: Aniket Ananda Kharat

# Roll no: 08

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[3]:


df = pd.read_csv('Wine.csv')


# In[4]:


df


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.dtypes


# In[8]:


df.shape


# In[9]:


for col in df.columns:
    print(f"Column {col} value counts:")
    print(df[col].value_counts())
    print("\n")


# In[10]:


df.isnull().sum()


# In[11]:


features = df.columns.difference(['Customer_Segment'])
x = df.loc[:, features].values
y = df.loc[:, ['Customer_Segment']].values


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 


# In[15]:


x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)


# In[16]:


x_train_std


# In[17]:


x_test_std


# In[18]:


from sklearn.decomposition import PCA


# In[19]:


pca = PCA(n_components=2)
x_train_transform = pca.fit_transform(x_train)
x_test_transform = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_


# In[20]:


explained_variance


# In[21]:


principal_df = pd.DataFrame(data=x_train_transform, columns=['PC1', 'PC2'])


# In[22]:


# Concatenate with the target variable
final_df = pd.concat([principal_df, df[['Customer_Segment']]], axis=1)


# In[23]:


final_df


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


clf = LogisticRegression()
clf.fit(x_train, y_train)


# In[26]:


y_pred = clf.predict(x_test)


# In[27]:


y_pred


# In[28]:


from sklearn.metrics import confusion_matrix


# In[29]:


cm = confusion_matrix(y_test,y_pred)


# In[30]:


cm


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:




