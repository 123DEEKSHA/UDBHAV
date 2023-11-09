#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('main1.csv')


# In[3]:


data1=data.copy()


# In[4]:


data1.head()


# x is the feauture y is the target. here status is target rest all feauture

# In[5]:


import warnings


# In[6]:


warnings.filterwarnings('ignore')


# In[7]:


data.head()


# In[8]:


data['Gender'].unique()


# In[9]:


data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})


# In[10]:


data.head()


# In[11]:


data['Qualification'].unique()


# In[12]:


data['Qualification'] = data['Qualification'].map({'Bachelor Degree':1, 'Diploma':2, 'PUC':3, 'BCA':4, 'PG':5, 'Phd':6})


# In[13]:


data.head()


# In[14]:


data['Origin'].unique()


# In[15]:


data['Origin'] = data['Origin'].map({'Urban':3, 'Remote':2, 'Rural':1, 'Semi urban':0})


# In[16]:


data.head()


# In[17]:


data['Company_type'].unique()


# In[18]:


data['Company_type'] = data['Company_type'].map({'Private limited company':6, 'Corporative':5, 'Partnership':4,
       'International company':3, 'Not yet working':2,
       'Nonprofit Organization':1, 'Private institution':0})


# In[19]:


data.head()


# In[20]:


data['Status'].unique()


# In[21]:


data['Status'] = data['Status'].map({'Employed':1, 'Unemployed':0})


# In[22]:


data.head()


# In[23]:


X=data.drop('Status',axis=1)
y=data['Status']


# In[24]:


X


# In[25]:


y


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[30]:


y_pred1 = lr.predict(X_test)


# In[31]:


y_pred1


# In[32]:


y_test


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


score1=accuracy_score(y_test,y_pred1)


# In[35]:


print(score1)


# In[36]:


data.columns


# In[37]:


data.head(2)


# In[38]:


new_data=pd.DataFrame({
    'Gender':0,
    'Qualification':2,
    'Passout':2017,
    'Origin':1,
    'Experience':0,
    'Company_type':2,
},index=[0])


# In[39]:


lr= LogisticRegression()
lr.fit(X,y)


# In[40]:


p=lr.predict(new_data)

if p==1:
    print('Employed')
else:
   
    print("unemployed")


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[42]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x='Experience',hue='Status',data=data1)
plt.subplot(1,2,2)
sns.countplot(x='Passout',hue='Status',data=data1)


# In[44]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x='Gender',hue='Status',data=data1)
plt.subplot(1,2,2)
sns.countplot(x='Origin',hue='Status',data=data1)


# In[45]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(x='Qualification',hue='Status',data=data1)
plt.subplot(1,2,2)
sns.countplot(x='Company_type',hue='Status',data=data1)


# In[ ]:




