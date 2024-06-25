#!/usr/bin/env python
# coding: utf-8

# SUPERVISED MACHINE LEARNING FINAL PROJECT
# 
# In this project we will take a Kaggle Dataset that shows student performance and factors that may contribute to it.
# 
# First we import the libraries we need.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv(r'Student_Performance.csv')


# After reading in the data we want to take a look to see what the data types are and if we need to clean the data in any way

# In[3]:


print(data.head())
print(data.info())
print(data.describe())


# In[4]:


print(data.isnull().sum())


# Our data has no NA values in any columns and looks very clean overall. For or Regression problem however we will want to convert 
# the extra curricular activities column  to a 1 for yes and a 0 for no.

# In[5]:


data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})


# we will now identify the features and the response

# In[6]:


X = data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']


# We will now split the data into training and test data and build our model.

# In[14]:


X = sm.add_constant(X)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


model = sm.OLS(y_train, X_train).fit()


# In[17]:


print(model.summary())


# looks like all features are significant however previous hours studied and previous scores have larger coefficients meaning
# that they will impact the student performance the most.

# Lets make some predictions now on the test data

# In[21]:


y_pred = model.predict(X_test)


# In[22]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# In[23]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.title('Actual vs Predicted Performance Index')
plt.show()


# The diagonal line suggests a very strong linear relationship between the predictors and the response. The predicted values are
# also really close to the actual values which means our linear regression model has very high accuracy.
# 
# There is also no pattern of curvature here which indicates that our choice of linear regression was appropriate for this
# particular dataset.
# 
# Our model has a high R2 score of 0.988, all of our predictors are significant with hours studied and previous scores contributing more to higher student performance. 

# In[ ]:




