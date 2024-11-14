#!/usr/bin/env python
# coding: utf-8

# # Name - Aniket Ananda Kharat
# # Batch - A , roll no - 2447008
# 

# ## Problem statement-Predict the price of the Uber ride from a given pickup point to the agreed drop-off location. Perform following tasks:
# 1. Pre-process the dataset.
# 2. Identify outliers.
# 3. Check the correlation.
# 4. Implement linear regression and ridge, Lasso regression models.
# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('uber.csv')


# In[4]:


df


# In[5]:


df.describe() 


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df.columns


# In[8]:


df.drop(columns=['Unnamed: 0', 'key'], axis=1, inplace=True)


# # Removing null values 

# In[9]:


df['dropoff_longitude'].fillna(df['dropoff_longitude'].mean(), inplace=True)
df['dropoff_latitude'].fillna(df['dropoff_latitude'].mean(), inplace=True)


# In[10]:


df.isnull().sum()


# In[11]:


df.pickup_datetime = pd.to_datetime(df.pickup_datetime)


# In[12]:


df


# In[13]:


df.drop('pickup_datetime', axis=1, inplace=True)


# In[14]:


df


# #  Feature extraction 

# In[15]:


df['distance'] = np.sqrt((df['dropoff_longitude'] - df['pickup_longitude'])**2 +
                         (df['dropoff_latitude'] - df['pickup_latitude'])**2)


# In[16]:


df


# # Correlation matrix

# In[24]:


import matplotlib.pyplot as plt


# In[26]:


correlation_matrix = df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# # removing outliers

# In[27]:


def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df


# In[28]:


for col in ['fare_amount', 'distance', 'passenger_count' ,'pickup_longitude' , 'pickup_latitude', 'dropoff_longitude' , 'dropoff_latitude']:
    df = remove_outlier(df, col)


# In[29]:


df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20)) 
plt.show()


# # features and target 

# In[30]:


X = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance']]
y = df['fare_amount']


# # Spliting dataset 

# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[32]:


# Scaling 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[33]:


X_train_scaled


# In[34]:


X_test_scaled


# # Lineaer Regression

# In[35]:


from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.metrics import mean_squared_error, r2_score


# In[36]:


linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)


# In[37]:


y_pred_linear = linear_reg.predict(X_test_scaled)


# In[38]:


linear_r2 = r2_score(y_test, y_pred_linear)
linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear))


# # Ridge Regression

# In[39]:


ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)


# In[40]:


y_pred_ridge = ridge_reg.predict(X_test_scaled)


# In[41]:


ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))


# # Lasso Regression

# In[42]:


lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train_scaled, y_train)


# In[43]:


y_pred_lasso = lasso_reg.predict(X_test_scaled)


# In[44]:


lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))


# # Comparing results

# In[45]:


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'R2 Score': [linear_r2, ridge_r2, lasso_r2],
    'RMSE': [linear_rmse, ridge_rmse, lasso_rmse]
})


# In[46]:


print(results)


# In[ ]:





# In[ ]:




