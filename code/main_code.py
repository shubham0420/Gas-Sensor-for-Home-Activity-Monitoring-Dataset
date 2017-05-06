
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# In[51]:

dataset = pd.read_csv('data/HT_Sensor_dataset.dat',sep = '  ',header = None,engine='python')
dataset.columns = ['id','time','R1','R2','R3','R4','R5','R6','R7','R8','Temp.','Humidity']
dataset.set_index('id',inplace = True)
dataset.head()


# In[55]:

output = pd.read_csv('data/HT_Sensor_metadata.csv',sep = '\t',header = None)
output.columns = ['id','date','class','t0','dt']
output.head()


# # Joining Dataset
# 
# We can see that the two dataset are for same experiment. So, we need to join the two datasets on id. 

# In[57]:

dataset = dataset.join(output,how = 'inner')
dataset.set_index(np.arange(dataset.shape[0]),inplace = True)
dataset['time']  += dataset['t0']
dataset.drop(['t0'],axis = 1,inplace=True)
dataset.head()


# Now, we are going to see the plots of reading of the sensors with time.  
# The graph shown represent day 17 reading.  
# As we can see from the graphs  sensor R7 shows minimum reading and sensor R1 has maximum reading.  
# The readings for rest of the days will be similar to the readings of plots shown below if the temprature and humidity are similar.

# In[59]:

fig, axes = plt.subplots(nrows=3, ncols=2)#, sharex=True, sharey=True)
fig.set_figheight(20)
fig.set_figwidth(25)
fig.subplots_adjust(hspace=.5)

axes[0,0].plot(dataset.time[dataset.id == 3],dataset.R1[dataset.id == 3],c = 'red',linewidth = '2.0')
axes[0,0].set_title('R1 Vs Time')
axes[0,0].set_xlabel('Time(hour)')
axes[0,0].set_ylabel('R1_values(kilo ohm)')

axes[0,1].plot(dataset.time[dataset.id == 3],dataset.R2[dataset.id == 3],c = 'green',linewidth = '2.0')
axes[0,1].set_title('R2 Vs Time')
axes[0,0].set_xlabel('Time(hour)')
axes[0,0].set_ylabel('R2_values (kilo ohm)')


axes[1,0].plot(dataset.time[dataset.id == 3],dataset.R3[dataset.id == 3],c = 'orange',linewidth = '2.0',label = 'R3 (Sensor)')
#axes[1,0].set_title('R3 Vs Time')
axes[1,0].set_xlabel('Time(hour)')
axes[1,0].set_ylabel('R3_values (kilo ohm)')


axes[1,0].plot(dataset.time[dataset.id == 3],dataset.R4[dataset.id == 3],c = 'blue',linewidth = '2.0',label = 'R4')
axes[1,0].set_title('R4 and R3 Vs Time')
axes[1,0].set_xlabel('Time(hour)')
axes[1,0].set_ylabel('Reading (kilo ohm)')
axes[1,0].legend(loc = 4)

axes[1,1].plot(dataset.time[dataset.id == 3],dataset.R5[dataset.id == 3],c = 'pink',linewidth = '2.0')
axes[1,1].set_title('R5 Vs Time')
axes[1,1].set_xlabel('Time(hour)')
axes[1,1].set_ylabel('R5_values (kilo ohm)')
 

axes[2,0].plot(dataset.time[dataset.id == 3],dataset.R6[dataset.id == 3],c = 'violet',linewidth = '2.0',label = 'R6')
#axes[2,0].set_title('R6 Vs Time')
axes[2,0].set_xlabel('Time(hour)')
axes[2,0].set_ylabel('R6_values (kilo ohm)')


axes[2,0].plot(dataset.time[dataset.id == 3],dataset.R7[dataset.id == 3],c = 'black',linewidth = '2.0',label ='R7')
axes[2,0].set_title('R7 and R6 Vs Time')
axes[2,0].set_xlabel('Time(hour)')
axes[2,0].set_ylabel('Reading (kilo ohm)')
axes[2,0].legend()

axes[2,1].plot(dataset.time[dataset.id == 3],dataset.R8[dataset.id == 3],c = 'brown',linewidth = '2.0')
axes[2,1].set_title('R8 Vs Time')
axes[2,1].set_xlabel('Time(hour)')
axes[2,1].set_ylabel('R8_values (kilo ohm)')
plt.suptitle('Sensor Reading on Day 3')
pl.savefig("Graph1.png", dpi=300)


# Now, the above reading will be similar for all days if the Humidity and Temprature are similar.  
# Let us plot the Humidity and Temprature vs Time.

# In[60]:

fig, axes = plt.subplots(nrows=1, ncols=2)#, sharex=True, sharey=True)
fig.set_figheight(5)
fig.set_figwidth(20)
fig.subplots_adjust(hspace=.5)

axes[0].plot(dataset.time[dataset.id == 17],dataset['Temp.'][dataset.id == 17],c = 'r')
axes[0].set_title('R1 Vs Temp')
axes[0].set_xlabel('Time (hour)')
axes[0].set_ylabel('Temprature (C)')
axes[1].plot(dataset.time[dataset.id == 17],dataset.Humidity[dataset.id == 17],c = 'green')
axes[1].set_title('R2 Vs Humidity')
axes[1].set_xlabel('Humidity (%)')
plt.suptitle('Temprature Reading on Day 3')
pl.savefig("Graph2.png", dpi=300)


# In[7]:

dataset.corr()


# In[8]:

dataset.corr() > 0.98


# In[17]:

xtrain_dataframe = pd.DataFrame(xtrain)
ytrain_dataframe = pd.DataFrame(ytrain)
xtest_dataframe = pd.DataFrame(xtest)
ytest_dataframe = pd.DataFrame(ytest)
xtrain_dataframe.columns = [u'R1',u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']
ytrain_dataframe.columns = ['class']
xtest_dataframe.columns = [u'R1',u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']
ytest_dataframe.columns = ['class']
res = sm.RLM(ytrain_dataframe, xtrain_dataframe).fit()
res.summary()


# When you perform a hypothesis test in statistics, a p-value helps you determine the significance of your results. Hypothesis tests are used to test the validity of a claim that is made about a population. This claim that’s on trial, in essence, is called the null hypothesis.
# 
# The alternative hypothesis is the one you would believe if the null hypothesis is concluded to be untrue. The evidence in the trial is your data and the statistics that go along with it. All hypothesis tests ultimately use a p-value to weigh the strength of the evidence (what the data are telling you about the population). The p-value is a number between 0 and 1 and interpreted in the following way:
# 
# 1. A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
# 
# 2. A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis.
# 
# 3. p-values very close to the cutoff (0.05) are considered to be marginal (could go either way). Always report the p-value so your readers can draw their own conclusions.
# 
# 
# #### P value of R1 is too much which means that this variable donot have affect on the model. Hence we can remove this variable from our model.

# In[8]:

xtrain,xtest,ytrain,ytest = train_test_split(dataset[[u'R2',u'R3',u'R4',u'R5',u'R6',u'R7',u'R8',u'Temp.',u'Humidity']].values,dataset['class'].values,train_size = 0.7)


# In[9]:

for i in range(ytrain.shape[0]):
    if(ytrain[i] == 'background'):
        ytrain[i] = 0
    elif(ytrain[i] == 'banana'):
        ytrain[i] = 1
    else:
        ytrain[i] = 2
        
for i in range(ytest.shape[0]):
    if(ytest[i] == 'background'):
        ytest[i] = 0
    elif(ytest[i] == 'banana'):
        ytest[i] = 1
    else:
        ytest[i] = 2
        
ytrain = ytrain.astype('int64')
ytest = ytest.astype('int64')


# # Logistic Regression

# Logistic Regression is a method for classification. We need to classify between the Wine, Banana and background. Logistic regression is a Linear classifier.   
# We need to regularize Logistic regression to prevent overfitting of the model.

# ## Cross Validation
# 
# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set X_test, y_test. Note that the word “experiment” is not intended to denote academic use only, because even in commercial settings machine learning usually starts out experimentally.
# 
# We did 11 fold cross validation

# ### Regularization
# 
# Regularization, in mathematics and statistics and particularly in the fields of machine learning and inverse problems, is a process of introducing additional information in order to solve an ill-posed problem or to prevent overfitting.

# In[46]:

from sklearn.cross_validation import KFold
Cs = [0.001,0.01,0.1,1,10,100]
count = 0
score_test = []
score_train = []
for c in Cs:
    score1 = []
    score2 = []
    est = LogisticRegression(C=c,n_jobs= 4)
    for itrain,itest in KFold(xtrain.shape[0],11):
        est.fit(xtrain[itrain],ytrain[itrain])
        score1.append(accuracy_score(est.predict(xtrain[itest]),ytrain[itest]))
        score2.append(accuracy_score(est.predict(xtrain[itrain]),ytrain[itrain]))
    score_test.append(np.mean(score1))
    score_train.append(np.mean(score2))


# In[72]:

plt.plot([0.001,0.01,0.1,1,10,100],score_train,'o-',label = 'training_score')
plt.plot([0.001,0.01,0.1,1,10,100],score_test,'o-',label = 'testing_score')
plt.xscale('log')
plt.legend(loc = 4)
plt.xlabel('Values of Regularization Parameter')
plt.ylabel('Accuracy Score')
plt.axhline(y=score_train[1],c = 'black')
plt.title('Prediction Acurracy of Logistic Regression')
pl.savefig("Graph3.png", dpi=300)


# We can see that the accuracy of both training and testing set is good at Cs = 0.1

# In[14]:

est = LogisticRegression(C = Cs[np.argmax(score)],n_jobs = 4)
est.fit(xtrain,ytrain)
ypred = est.predict(xtest)


# In[15]:

confusion_matrix(ytest, ypred)


# Acurracy :- 63%.  
# From this graph we can see that the regularization is causing the model to underfit. It is because the model is not linearly seprable. So, we need to add feature to increase the accuracy of the model.

# # Support Vector Machine

# In[12]:

from sklearn.svm import SVC

C_2d_range = [1e-2]
gamma_2d_range = [1e-1]
classifiers = []
for C in C_2d_range:
    for gamma in gamma_2d_range:
        clf = SVC(C=C, gamma=gamma)
        clf.fit(xtrain, ytrain)
        classifiers.append((C, gamma, clf))


# In[14]:

accuracy_score(clf.predict(xtest),ytest)


# As we can see the accuray is much high for training set. Let us check the accuracy of SVM for training set

# In[16]:

accuracy_score(clf.predict(xtrain),ytrain)


# #### Hence we have approached our desired accuracy which is better than the paper we followed. 
# Training Set Accuracy 96.22%.  
# Testing Set Accuracy 96.18%

# # We achieve more accuracy than the paper.
