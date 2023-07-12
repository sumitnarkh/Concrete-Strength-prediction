#!/usr/bin/env python
# coding: utf-8
Problem Statement:- How strong will be the concrete mixture? Can you estimate it while creating it? A seasoned civil engineer will know the winning mixture by heart! He/she will understand what should be the right amount of water, ash, cement etc. should be mixed in order to create a high strength concrete mixture.

Data Description:
 - CementComponent: How much cement is mixed
 - BlastFurnaceSlag: How much Blast Furnace Slag is mixed
 - FlyAshComponent: How much FlyAsh is mixed
 - WaterComponent: How much water is mixed
 - SuperplasticizerComponent: How much Super plasticizer is mixed
 - CoarseAggregateComponent: How much Coarse Aggregate is mixed
 - FineAggregateComponent: How much Coarse Aggregate is mixed
 - AgeInDays: How many days it was left dry
 - Strength: What was the final strength of concrete.
# In[1]:


# import libraries
import pandas as pd
import numpy as np


# In[35]:


data = pd.read_csv("ConcreteStrengthData.csv")


# In[36]:


data.head()


# In[37]:


data.shape


# In[38]:


# Removing duplicates row if any
data = data.drop_duplicates()


# In[39]:


data.shape

Defining the problem statement:

 Create a ML model which can predict the Strength of concrete
- Target Variable: Strength
- Predictors: water, cement, ash, days to dry etc.
# In[40]:


data.head()


# In[41]:


data.info()


# In[42]:


data.describe()


# In[43]:


data.nunique()

let's look at the distribution of Target variable:

 - If target variable's distribution is too skewed then the predictive modeling will not be possible.
 - Bell curve is desirable but slightly positive skew or negative skew is also fine
 - When performing Regression, make sure the histogram looks like a bell curve or slight skewed version of it. Otherwise it        impacts the Machine Learning algorithms ability to learn all the scenarios.
# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


data['Strength'].hist()

The data distribution of the target variable is statisfatory to proceed further.  
# In[45]:


## Exploratory Data Analysis
  #catagorical variables: Bar Plot
  #Continuous variables: Histogram

# Plotting histogram for multiple columns together

data.hist(['CementComponent ', 'BlastFurnaceSlag', 'FlyAshComponent',
                'WaterComponent', 'SuperplasticizerComponent','CoarseAggregateComponent', 
                           'FineAggregateComponent', 'AgeInDays'], figsize=(18,10))

Histograms shows us the data distribution for a single continuous variable.

The X-axis shows the range of values and Y-axis represent the number of values in that range. For example, in the above histogram of "AgeInDays", there are around 800 rows in data that has a value between 0 to 25.

The ideal outcome for histogram is a bell curve or slightly skewed bell curve. If there is too much skewness, then outlier treatment should be done and the column should be re-examined, if that also does not solve the problem then only reject the column.
# In[46]:


data.isnull().sum()


# In[ ]:


## Outlier Treatement ## 


# In[22]:


ContinuousCols=['CementComponent ', 'BlastFurnaceSlag', 'FlyAshComponent',
                'WaterComponent', 'SuperplasticizerComponent','CoarseAggregateComponent', 
                           'FineAggregateComponent', 'AgeInDays']    

# Plotting scatter chart for each predictor vs the target variables
for predictor in ContinuousCols:
    df.plot.scatter(x=predictor, y='Strength', figsize=(10,5), title=predictor+" VS "+ 'Strength')


# In[47]:


# Calculating correlation matrix
ContinuousCols=['Strength','CementComponent ', 'BlastFurnaceSlag', 'FlyAshComponent',
                'WaterComponent', 'SuperplasticizerComponent','CoarseAggregateComponent', 
                           'FineAggregateComponent', 'AgeInDays']

# Creating the correlation matrix
CorrelationData=data[ContinuousCols].corr()
CorrelationData


# In[26]:


# Filtering only those columns where absolute correlation > 0.5 with Target Variable
# reduce the 0.5 threshold if no variable is selected
CorrelationData['Strength'][abs(CorrelationData['Strength']) > 0.3 ]

Final selected Continuous columns:
'CementComponent','SuperplasticizerComponent','AgeInDays'
# In[51]:


## Data preprocessing for machine learning  ## 

#### Converting the nominal variable to numeric using get_dummies()

# Treating all the nominal variables at once using dummy variables
data_Numeric=pd.get_dummies(df)


# In[52]:


data_Numeric.head()


# In[53]:


## Splitting the data into training and testing sample ## 
#Separate Target Variable and Predictor Variables
TargetVariable='Strength'
Predictors=['CementComponent ', 'SuperplasticizerComponent', 'AgeInDays']

X=data_Numeric[Predictors].values
y=data_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)


# In[54]:


# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[55]:


## Random Forest ## 

from sklearn.ensemble import RandomForestRegressor
RegModel = RandomForestRegressor(max_depth=5, n_estimators=100)


# In[56]:


# Creating the model on Training Data
RF=RegModel.fit(X_train,y_train)
prediction=RF.predict(X_test)


# In[57]:


from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, RF.predict(X_train)))


# In[58]:


# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(RF.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# In[59]:


print('\n##### Model Validation and Accuracy Calculations ##########')

# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())


# In[60]:


# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['Strength']-TestingDataResults['PredictedStrength']))/TestingDataResults['Strength'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)


# In[61]:


# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)


# In[62]:


# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[63]:


## AdaBoost ## 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[64]:


DTR=DecisionTreeRegressor(max_depth=10)
RegModel = AdaBoostRegressor(n_estimators=100, base_estimator=DTR ,learning_rate=0.04)


# In[65]:


# Creating the model on Training Data
AB=RegModel.fit(X_train,y_train)
prediction=AB.predict(X_test)


# In[66]:


from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, AB.predict(X_train)))


# In[67]:


# Plotting the feature importance for Top 10 most important columns
get_ipython().run_line_magic('matplotlib', 'inline')
feature_importances = pd.Series(AB.feature_importances_, index=Predictors)
feature_importances.nlargest(10).plot(kind='barh')


# In[68]:


# Printing some sample values of prediction
TestingDataResults=pd.DataFrame(data=X_test, columns=Predictors)
TestingDataResults[TargetVariable]=y_test
TestingDataResults[('Predicted'+TargetVariable)]=np.round(prediction)

# Printing sample prediction values
print(TestingDataResults[[TargetVariable,'Predicted'+TargetVariable]].head())


# In[69]:


# Calculating the error for each row
TestingDataResults['APE']=100 * ((abs(
  TestingDataResults['Strength']-TestingDataResults['PredictedStrength']))/TestingDataResults['Strength'])

MAPE=np.mean(TestingDataResults['APE'])
MedianMAPE=np.median(TestingDataResults['APE'])

Accuracy =100 - MAPE
MedianAccuracy=100- MedianMAPE
print('Mean Accuracy on test data:', Accuracy) # Can be negative sometimes due to outlier
print('Median Accuracy on test data:', MedianAccuracy)


# In[70]:


#Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def Accuracy_Score(orig,pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)

# Custom Scoring MAPE calculation
from sklearn.metrics import make_scorer
custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)


# In[71]:


# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values=cross_val_score(RegModel, X , y, cv=10, scoring=custom_Scoring)
print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# In[ ]:




