#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns


# In[6]:


# Loading the dataset
Ocean_df = pd.read_csv('bottle.csv')


# In[7]:


# Getting the dimensions of the dataset
Ocean_df.shape


# In[8]:


# Return the description of the data in the DataFrame
Ocean_df.describe


# In[9]:


# Creating dataframe for the two columns that will be used
Ocean = Ocean_df[['T_degC','Salnty']]


# In[10]:


Ocean.shape


# In[11]:


# Checking for missing values per column
Ocean.isnull().sum(axis=0)


# In[12]:


# Detecting the various formats that can be represented as missing values
missing_values = ["n/a", "na", "--"]


# In[13]:


# Getting rid of missing values from dataframe
Oceans = pd.read_csv("bottle.csv", na_values = missing_values)


# In[14]:


OceanDF = Oceans[['T_degC','Salnty']]


# In[15]:


# Calculate the sum of elements in each column
OceanDF.isnull().sum(axis=0)


# In[16]:


# Description of the column temperature

print(OceanDF["T_degC"].describe())
plt.figure(figsize=(9, 8))
sns.distplot(OceanDF["T_degC"], color='r', bins=100, hist_kws={'alpha': 0.4}).set(xlabel='Ocean Temperature (°C)',title='Ocean Temperature (°C)');


# In[17]:


# Description of the column salinity
print(OceanDF["Salnty"].describe())
plt.figure(figsize=(9, 8))
sns.distplot(OceanDF["Salnty"], color='g', bins=100, hist_kws={'alpha': 0.4}).set(xlabel='Ocean salinity in g of salt per kg of water (g/kg)',title='Ocean salinity');


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(10,5), 'figure.dpi':100})
# plot boxplot
OceanDF["T_degC"].plot.box(sym='+')
plt.title('Ocean Temperature (°C)');


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.figsize':(10,5), 'figure.dpi':100})
# plot boxplot
OceanDF["Salnty"].plot.box(sym='+')
plt.title('Ocean salinity in g of salt per kg of water (g/kg)');


# In[20]:


# Scatter plot for Ocean salinity against temperature
OceanDF.plot.scatter(x="T_degC", y="Salnty",color='green',alpha=0.6,edgecolors="white",linewidth=1)
plt.title('Salinity against Ocean Temperature')
plt.show()


# In[21]:


# Histogram comparison for Ocean temperature and salinity
xweights = 100 * np.ones_like(OceanDF["T_degC"]) / OceanDF["T_degC"].size
yweights = 100 * np.ones_like(OceanDF["Salnty"]) /OceanDF["Salnty"].size

fig, ax = plt.subplots()
ax.hist(OceanDF["T_degC"], weights=xweights,label= 'Ocean Temperature (°C)', color='blue', alpha=0.5, edgecolor='black')
ax.hist(OceanDF["Salnty"], weights=yweights,label='Ocean salinity in g of salt per kg of water (g/kg)', color='salmon', alpha=0.5, edgecolor='black')

ax.set(title='Histogram Comparison', ylabel='% of Dataset in Bin')
ax.margins(0.05)
ax.set_ylim(bottom=0)
plt.legend()
plt.show()


# In[22]:


Salinity = OceanDF["Salnty"].values[0:800000]
Temperature = OceanDF["T_degC"].values[0:800000]
plt.scatter(Temperature,Salinity)
plt.show()


# In[23]:


Sal_vector = Salinity.reshape(-1,1)


# In[28]:


model = LinearRegression().fit(Sal_vector, Temperature)


# In[26]:


prediction = model.predict(depth_vector)


# In[27]:


plt.scatter(Temperature,Salinity)
plt.plot(Salinity, prediction, color="red")
plt.title('Salinity against Ocean Temperature linear regression before training')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Temperature, Sal_vector, train_size=.8, test_size=.2)


# In[ ]:


print(f"X_train shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"X_train shape {X_test.shape}")
print(f"y_train shape {y_test.shape}")


# In[ ]:


# Plot the training data
plt.scatter(X_train, y_train, color='red')
plt.xlabel('Depth in Meters')
plt.ylabel('Temperature in C')
plt.title('Training data')
plt.show()


# In[ ]:


plt.scatter(X_test, y_test, color='blue')
plt.xlabel('Depth in Meters')
plt.ylabel('Temperature in C')
plt.title('Test data')
plt.show()


# In[ ]:


X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)


# In[ ]:


lm = LinearRegression()
lm.fit(X_train, y_train)


# In[ ]:


print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)


# In[ ]:


X_test = X_test.reshape(-1,1)


# In[ ]:


y_predict = lm.predict(X_test)


# In[ ]:


print(f"Train accuracy {round(lm.score(X_train,y_train)*100,2)} %")


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))


# In[ ]:




