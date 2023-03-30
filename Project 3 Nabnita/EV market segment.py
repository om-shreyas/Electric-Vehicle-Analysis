#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().system('pip install bioinfokit')


# In[2]:


data =pd.read_csv("EV survery.csv")


# In[4]:


data.shape
data.head()
data.dtypes
data.info()
data.isnull().sum()


# In[5]:


data.head()


# In[6]:


data['Gender'].value_counts()
data['Age'].value_counts()


# In[8]:


#EXPLORING DATA

#Customer segmentation - based on socio-demographs (Age & Gender)

#Gender
labels = ['Female', 'Male']


# In[15]:


data.groupby('Gender').size().plot(kind='bar')


# In[16]:


#Age
plt.rcParams['figure.figsize'] = (25, 8)
f = sns.countplot(x=data['Age'],palette = 'hsv')
f.bar_label(f.containers[0])
plt.title('Age distribution of customers')
plt.show()


# In[17]:


data.isnull().any()


# In[18]:


data['Profession'].unique()


# In[20]:


data['Location'].unique()


# In[21]:


data['Type of location'].unique()


# In[22]:


data['Model'].unique()


# In[23]:


data['Segment'].unique()


# In[26]:


data['Environmet'].unique()


# In[27]:


data['PowerTrain'].unique()


# In[28]:


data['Price'].unique()


# In[29]:


data['Choice'].unique()


# In[30]:


data1=data.drop(['Model','Segment'],axis=1)


# In[31]:


data1.head()


# In[32]:


sns.pairplot(data1)
plt.show()


# In[33]:


plt.figure(figsize=[9,7])
data1['Environmet'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[34]:


plt.figure(figsize=[9,7])
data1['Annual Income'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[35]:


plt.figure(figsize=[9,7])
data1['Price'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[46]:


plt.figure(figsize=[9,7])
data['Model'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[49]:


plt.figure(figsize=[20,18])
data['Segment'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[36]:


plt.figure(figsize=[9,7])
data1['Choice'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[37]:


plt.figure(figsize=[9,7])
data1['Budget'].value_counts().plot.pie(autopct='%.0f%%')
plt.show()


# In[38]:


data['Annual Income'].unique()


# In[61]:


data['Location'].unique()


# In[39]:


data['Annual Income'] = data['Annual Income'].replace(to_replace='5 lpa',value=1)
data['Annual Income'] = data['Annual Income'].replace(to_replace='20 to 30 lpa',value=2)
data['Annual Income'] = data['Annual Income'].replace(to_replace='5 to10 lpa',value=3)
data['Annual Income'] = data['Annual Income'].replace(to_replace='10 to 20 lpa',value=4)
data['Annual Income'] = data['Annual Income'].replace(to_replace='50 + lpa',value=5)
data['Annual Income'] = data['Annual Income'].replace(to_replace='30 to 40 lpa',value=6)


# In[40]:


data['Type of location'] = data['Type of location'].replace(to_replace='Major City',value=1)
data['Type of location'] = data['Type of location'].replace(to_replace='Minor City',value=2)
data['Type of location'] = data['Type of location'].replace(to_replace='Town',value=3)
data['Type of location'] = data['Type of location'].replace(to_replace='Village',value=4)


# In[43]:


data['PowerTrain'] = data['PowerTrain'].replace(to_replace='Lack of charging options',value=1)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='Range in terms of distance',value=2)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='Affordability',value=3)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='Lack of design options',value=4)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='I dont have any concerns',value=5)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace="EVs don't produce sound",value=6)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='Performance',value=7)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace="It doesn't make sound while moving. It may cause accidents",value=8)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='I dont like how soundless they they dont have a natural strong feeling to it',value=9)
data['PowerTrain'] = data['PowerTrain'].replace(to_replace='The initial carbon footprint that it generates',value=10)


# In[50]:


data['Price'] = data['Price'].replace(to_replace='1-2 lakhs',value=2)
data['Price'] = data['Price'].replace(to_replace='5-10 lakhs',value=5)
data['Price'] = data['Price'].replace(to_replace='3-5 lakhs',value=3)
data['Price'] = data['Price'].replace(to_replace="I wouldn't like to pay any extra price",value=0)
data['Price'] = data['Price'].replace(to_replace='10 lakhs +',value=10)


# In[51]:


data['Choice'] = data['Choice'].replace(to_replace='Negligible Running cost',value=0)
data['Choice'] = data['Choice'].replace(to_replace='I do not prefer an electric car',value=1)
data['Choice'] = data['Choice'].replace(to_replace='Enviroment Friendly',value=10)
data['Choice'] = data['Choice'].replace(to_replace='Overall better performance than many gas cars.',value=9)


# In[52]:


data['Model'] = data['Model'].replace(to_replace='SUV',value=41)
data['Model'] = data['Model'].replace(to_replace='Sedan',value=35)
data['Model'] = data['Model'].replace(to_replace='MPV (Multi purpose vehicle)',value=5)
data['Model'] = data['Model'].replace(to_replace= 'Hatch back',value=10)
data['Model'] = data['Model'].replace(to_replace='Sports',value=9)


# In[53]:


data['Segment'] = data['Segment'].replace(to_replace='Performance',value=18)
data['Segment'] = data['Segment'].replace(to_replace='Affordability',value=11)
data['Segment'] = data['Segment'].replace(to_replace='Luxury',value=8)
data['Segment'] = data['Segment'].replace(to_replace= 'Style',value=6)
data['Segment'] = data['Segment'].replace(to_replace='Running Cost',value=5)


# In[54]:


data


# In[55]:


data['PowerTrain'].describe()


# In[56]:


data_bool = data.drop("PowerTrain",axis=1)
data_bool = data_bool.drop("Gender",axis=1)
data_bool = data_bool.drop("Age",axis=1)
data_bool = data_bool.drop("Model",axis=1)


# In[57]:


for i in data_bool:
    print(data_bool[i].describe())
    print()


# In[58]:


sns.set(rc = {'figure.figsize':(15,8)})
sns.heatmap(data.corr(),annot=True,cmap='BuGn')


# In[66]:


from sklearn.decomposition import PCA


# In[71]:


from sklearn.cluster import KMeans


# In[ ]:




