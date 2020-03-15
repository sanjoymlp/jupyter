#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


from sklearn.linear_model import LinearRegression


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns


# In[4]:


p=pd.read_csv(r"C:\Users\sanjoy\Downloads\real_estate_price_size_year_view.csv")


# In[5]:


spp=pd.read_csv(r"C:\Users\sanjoy\Downloads\secondcar.csv")
sppp=pd.read_csv(r"C:\Users\sanjoy\Downloads\car_sales.csv")
self=pd.read_csv(r"C:\Users\sanjoy\Downloads\salary_data.csv")


# In[6]:


self.head()


# In[7]:


real_x=self[['milestraveled','numdeliveris','gasprice']]


# In[8]:


real_y=self[['traveltime']]
real_y


# In[9]:


training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.3,random_state=0)


# In[10]:


Lin=LinearRegression()
Lin.fit(training_x, training_y)
pred_y=Lin.predict(testing_x)


# In[11]:


pred_y


# In[12]:


testing_y


# In[13]:


Lin.coef_


# In[14]:


Lin.intercept_


# In[62]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(testing_y, pred_y))
print('Mean Squared Error:', metrics.mean_squared_error(testing_y, pred_y))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testing_y,pred_y)))


# In[ ]:




