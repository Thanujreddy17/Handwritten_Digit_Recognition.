#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale


# In[2]:


digits = pd.read_csv(r'train.csv')


# In[3]:


digits.shape


# In[4]:


#Since the dataset is too large we'll be working on only 20% of the data.
data = digits.head(10000)


# In[5]:


print(data.shape)
data.head()


# In[6]:


#splitting into X and Y
x = data.drop('label', axis=1)
Y_train = data['label']
x.shape


# In[7]:


X_train = scale(x)


# In[8]:


test = digits.iloc[30000:34201,:]
X_test = test.drop('label', axis=1)
X_test = scale(X_test)
Y_test = test['label']
X_test.shape


# In[9]:


#linear-model
model_linear = SVC(kernel = 'linear')
model_linear.fit(X_train,Y_train)


# In[10]:


#predict
Y_pred = model_linear.predict(X_test)


# In[11]:


#accuracy
print("accuracy:", metrics.accuracy_score(y_true=Y_test, y_pred=Y_pred), "\n")

# confusion matrix
print(metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred))


# In[12]:


#non-linear model
model_non_linear = SVC(kernel = 'rbf')
model_non_linear.fit(X_train,Y_train)


# In[13]:


#predict
Y_pred1 = model_non_linear.predict(X_test)


# In[14]:


#accuracy
print("accuracy:", metrics.accuracy_score(y_true=Y_test, y_pred=Y_pred1), "\n")

# confusion matrix
print(metrics.confusion_matrix(y_true=Y_test, y_pred=Y_pred1))


# #### _<font color='red'>GridSearch: Hyperparameter Tuning</font>_

# In[25]:


folds = KFold(n_splits=5, shuffle=True, random_state=100)

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4], 
                  'C': [1, 10, 100, 1000]}]

# specifying model
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, param_grid = hyper_params,
                        scoring= 'accuracy',cv = folds,verbose = 1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train, Y_train)


# In[26]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[27]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]
plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]
plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]
plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# In[28]:


# printing the optimal accuracy score and hyperparameters
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_


# In[29]:


print(best_hyperparams)


# In[31]:


model = SVC(C=10, gamma=0.001, kernel="rbf")

#Fitting the model.
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#Returning 'accuracy'.
print("accuracy", metrics.accuracy_score(Y_test, Y_pred), "\n")
print(metrics.confusion_matrix(Y_test, Y_pred), "\n")

