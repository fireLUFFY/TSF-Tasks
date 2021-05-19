#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION 
# 
# ## DOMAIN: Prediction Using Unsupervised ML
# 
# ### Predict the optimum numbers of clusters & represent it visually

# ### BY: Rahul Kumar Sethi
# ### National Institute of Technology - Rourkela

# #### Importing the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# #### Importing Iris dataset by downloading the file from the url( https://bit.ly/3kXTdox )
# 
# #### Checking data for any anomalies

# In[2]:


Data=pd.read_csv('C:/Users/rahulsethi/Documents/ML_DS/Iris.csv')
print('Data imported')
Data.head(10)


# In[3]:


Data.info()


# In[4]:


Data.describe()


# #### Determining the optimum number of clusters for K Means. Using " The Elbow " method, from the plot, the optimum K value occurs at the elbow. Here, the sum of cluster of squares(W) doesn't decrease significantly. 
# #### From the plot, the optimum cluster occurs at around 3. 

# In[5]:


x=Data.iloc[:,[0,1,2,3]].values
W=[]

for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    W.append(km.inertia_)
    
plt.plot(range(1,11),W)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('W')
plt.show()


# #### Using the optimum value, creating the K-Means classifier for the dataset

# In[6]:


km=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_km=km.fit_predict(x)


# #### Visualising the clusters and plotting the centroids

# In[7]:


plt.scatter(x[y_km==0,0],x[y_km==0,1],s=90,c='Violet',label='Iris-setosa')
plt.scatter(x[y_km==1,0],x[y_km==1,1],s=90,c='cyan',label='Iris-versicolour')
plt.scatter(x[y_km==2,0],x[y_km==2,1],s=90,c='orange',label='Iris-virginica')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=90,c='blue',label='Centroids')
plt.legend()

