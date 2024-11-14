#!/usr/bin/env python
# coding: utf-8

# # Name - Aniket Kharat
# # Roll no - 2447008 , batch - A

# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# In[3]:


df = pd.read_csv('car_evaluation.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[ ]:





# In[7]:


for col in df.columns:
    print(f"Column {df[col].value_counts()} value count")
    print("\n")


# In[8]:


le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])


# In[9]:


X = df.drop(columns=['unacc'])  # Features
y = df['unacc']  # Target (safety classification)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
#n_estimators = 100 means 100 decision trees 


# In[12]:


y_pred = clf.predict(X_test)


# In[13]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[17]:


print("Classification report")
print(classification_report(y_test,y_pred))


# In[14]:


feature_importance = clf.feature_importances_
feature_names = X.columns


# In[15]:


plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance in Predicting Car Safety')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()


# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , classification_report
from sklearn.ensemble import RandomForestClassifier


# In[4]:


df = pd.read_csv('car_evaluation.csv')


# In[5]:


df


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df.shape


# In[11]:


for col in df.columns:
    print(f'Column -> {df[col].value_counts()}value count')
    print('\n')


# In[16]:


le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])


# In[ ]:





# In[ ]:





# In[20]:


x = df.drop(columns=['unacc']) # features 
y = df['unacc']


# In[21]:


x


# In[22]:


y


# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)


# In[26]:


clf_ran  = RandomForestClassifier(n_estimators = 100 , random_state = 42)


# In[28]:


clf_ran.fit(x_train,y_train)


# In[29]:


Y_pred = clf_ran.predict(x_test)


# In[30]:


acc = accuracy_score(y_test,Y_pred)
print(f'accuracy is -> {acc*100:.2f}%')


# In[31]:


print('classification report')
print(classification_report(y_test,Y_pred))


# In[34]:


feature_imp = clf_ran.feature_importances_
features_name = x.columns


# In[39]:


features_name


# In[41]:


import seaborn as sns


# In[43]:


plt.figure(figsize=(8,8))
sns.barplot(x = feature_imp , y = features_name)
plt.title('features importance ')
plt.plot()
plt.show()


# In[ ]:




