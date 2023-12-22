#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


df = pd.read_csv("homeprices.csv")
df


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sq ft)")
plt.ylabel("prices(US$)")
plt.scatter(df.area,df.price,color='orange',marker='*')


# In[7]:


reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.price)


# In[10]:


reg.predict([[3300]])


# In[12]:


reg.coef_


# In[13]:


reg.intercept_


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("area(sq ft)")
plt.ylabel("prices(US$)")
plt.scatter(df.area,df.price,color='orange',marker='*')
plt.plot(df.area,reg.predict(df[['area']]),color='red')


# In[15]:


d=pd.read_csv("areas.csv")
d.head(3)


# In[17]:


p=reg.predict(d)


# In[18]:


d['prices']=p


# In[22]:


d.to_csv("prediction.csv",index=False)


# In[ ]:




