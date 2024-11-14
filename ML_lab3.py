#!/usr/bin/env python
# coding: utf-8

# # Name - Aniket Kharat
# # Roll no - 2447008 , Batch - A

# ### Problem statement -  Implementation of Support Vector Machines (SVM) for classifying images of handwritten digits into their respective numerical classes (0 to 9)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


digits = datasets.load_digits()


# In[5]:


x = digits.data  
y = digits.target 


# In[6]:


x


# In[7]:


y


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[13]:


X_train_std


# In[14]:


X_test_std


# In[15]:


svm_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
#what is kernel and 'rbf' , gamma = 'scale ' , C = 1.0
svm_rbf.fit(X_train, y_train)
y_rbf_pred = svm_rbf.predict(X_test)


# In[16]:


accuracy = accuracy_score(y_test, y_rbf_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_rbf_pred))


# In[20]:


for i in np.random.randint(0, len(X_test), 4): 
    two_d = np.reshape(X_test[i], (8, 8))  
    plt.title(f'Predicted label: {y_rbf_pred[i]}')
    plt.imshow(two_d, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()


# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Iris.csv')


# In[3]:


df


# In[5]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score , classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[6]:


digit = datasets.load_digits()


# In[7]:


digit


# In[9]:


x = digit.data
y = digit.target


# In[10]:


x


# In[11]:


y


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state=42)


# In[ ]:





# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


sc = StandardScaler()


# In[18]:


x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)


# In[19]:


x_train_std


# In[20]:


x_test_std


# In[21]:


svm_rbf = SVC(kernel = 'rbf', gamma ='scale',C= 1.0)
svm_rbf.fit(x_train,y_train)
y_rbf_pred = svm_rbf.predict(x_test)


# In[22]:


acc = accuracy_score(y_test,y_rbf_pred)


# In[25]:


print(f"accuracy:{acc*100:.2f}%")
print("classification report")
print(classification_report(y_test,y_rbf_pred))


# In[27]:


import matplotlib.pyplot as plt


# In[30]:


for i in np.random.randint(0,len(x_test),8):
    two_d=np.reshape(x_test[i],(8,8))
    plt.title(f'predicted digit:{y_rbf_pred[i]}')
    plt.imshow(two_d,cmap='gray',interpolation='nearest')
    plt.colorbar()
    plt.show()


# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score , classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[2]:


digits = datasets.load_digits()


# In[3]:


digits


# In[7]:


x=digits.data
y=digits.target


# In[8]:


x


# In[9]:


y


# In[14]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state=42)


# In[15]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[16]:


x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)


# In[17]:


x_train_std


# In[18]:


x_test_std


# In[19]:


svm_rbf = SVC(kernel = 'rbf' , gamma = 'scale' , C = 1.0)
svm_rbf.fit(x_train,y_train)
y_rbf_pred = svm_rbf.predict(x_test)


# In[20]:


from sklearn.metrics import accuracy_score , classification_report


# In[25]:


acc = accuracy_score(y_test,y_rbf_pred)
print(f'Accuracy is - {acc*100:.2f}% ')

import matplotlib.pyplot as plt


# In[26]:


print(classification_report(y_test,y_rbf_pred))


# In[28]:


for i in np.random.randint(0,len(x_test),8):
    two_d=np.reshape(x_test[i],(8,8))
    plt.title(f'predicted digit:{y_rbf_pred[i]}')
    plt.imshow(two_d,cmap='gray',interpolation='nearest')
    plt.colorbar()
    plt.show()


# In[ ]:




