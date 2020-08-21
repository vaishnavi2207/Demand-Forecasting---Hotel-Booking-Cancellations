#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv (r'C:\Users\vaish\OneDrive\Desktop\PYTHON PROJECTS\hotel_bookings.csv')
print (df)


# In[ ]:


get_ipython().system('pip install catboost')
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from sklearn.model_selection import train_test_split
import datetime
from datetime import timedelta
from sklearn.utils import resample
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.metrics import confusion_matrix,classification_report,recall_score,roc_auc_score,auc, confusion_matrix, roc_curve,accuracy_score
import sklearn 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV


# In[ ]:





# In[ ]:


#Data Preprocessing


# In[ ]:


perc_null=((hotel_df.isnull().sum()/hotel_df.shape[0])*100).sort_values(ascending=False)
perc_null


# In[ ]:


hotel_df =  hotel_df.drop(['agent','company'],axis = 1)


# In[ ]:


#Correlations


# In[ ]:



plt.figure(figsize=(16, 9)) 
corr_matrix = hotel_df.corr()
sns.heatmap(corr_matrix)
plt.show()


# In[ ]:


corr_with_cancel = corr_matrix["is_canceled"]
corr_with_cancel.abs().sort_values(ascending=False)[1:]


# In[ ]:


#MISSING VALUE IMPUTATION IN TRAIN AND TEST
#Before Imputing missing values, splitting into Train and test to avoid data leakage.


# In[ ]:


X_train, X_test = train_test_split(hotel_df, test_size = 0.30, random_state = 42)
X_train.shape, X_test.shape


# In[ ]:


def imputation(df):
  df["country"]=df["country"].fillna("Missing")
  df["children"]=df["children"].fillna(df["children"].mode()[0])
  return(df)


# In[ ]:



X_train = imputation(X_train)
X_test = imputation(X_test)


# In[42]:


hotel_df=pd.concat([X_train,X_test])
hotel_df.shape


# In[ ]:


#Feature Engineering


# In[43]:


def arrival_date(data):
    date=datetime.datetime.strptime(str(data["arrival_date_year"])+data["arrival_date_month"]+str(data["arrival_date_day_of_month"]),"%Y%B%d")
    return(date)

def family_or_not(data):
    if (data["adults"]>0) &((data["children"]>0)|(data["babies"]>0)):
        return(1)
    else:
        return(0)
        
def room_type_not_given(data):
    if data["reserved_room_type"]==data["assigned_room_type"]:
        return(0)
    else: 
        return(1)


# In[44]:


def feature_eng(data):
    data["arrival_date"]=data.apply(arrival_date,axis=1)
    data["arrival_date_weekday"]=data["arrival_date"].dt.weekday
    data["family_or_not"]=data.apply(family_or_not,axis=1)
    data["room_type_changes"]=data.apply(room_type_not_given,axis=1)

    data["Total_No_Of_Nights_Stayed"]=data["stays_in_weekend_nights"]+data["stays_in_week_nights"]
    data["Non_Refund_Flag"]=data["deposit_type"].apply(lambda x:1 if x=="Non Refund" else 0)
    return(data)


# In[45]:


hotel_df=feature_eng(hotel_df)


# In[46]:


# Dropping variables used to create new features and reservation_status as label depends on it
hotel_df = hotel_df.drop(['arrival_date_month','arrival_date_year','arrival_date_day_of_month',
                          'reservation_status','reservation_status_date'],axis= 1)


# In[ ]:


#ONE HOT ENCODING


# In[47]:


hotel_df=pd.get_dummies(hotel_df,dtype='int64',drop_first=True)


# In[48]:


hotel_df.shape


# In[ ]:


#Split into x and y


# In[49]:


y=hotel_df["is_canceled"]
X=hotel_df.drop("is_canceled",axis=1)


# In[50]:


#SPLIT INTO TRAIN AND TEST


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42,stratify = y)


# In[52]:


#y train - Imbalanced data set
y_train.value_counts()/y_train.shape[0]*100


# In[53]:



#y test
y_test.value_counts()/y_test.shape[0]*100


# In[54]:


#BALANCING THE DATA


# In[55]:



merged_xy=X_train.merge(y_train,left_index=True,right_index=True)
not_cancel=merged_xy[merged_xy['is_canceled']==0]
cancel=merged_xy[merged_xy['is_canceled']==1]
print(not_cancel.shape)
print(cancel.shape)


# In[56]:


not_cancel_sampled = resample(not_cancel,
                                replace = False, 
                                n_samples = len(cancel), 
                                random_state = 20)


# In[57]:


X_train = pd.concat([not_cancel_sampled, cancel])
X_train.shape


# In[58]:


y_train=X_train['is_canceled']
X_train = X_train.drop('is_canceled',axis = 1)


# In[59]:


X_train.shape,y_train.shape


# In[60]:


corr = X_train.corr()
upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
corr_threshold=0.90
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
to_drop


# In[61]:



X_train = X_train.drop(X_train[to_drop],axis=1)
X_test = X_test.drop(X_test[to_drop], axis=1)


# In[62]:


X_train = X_train.drop('arrival_date',axis=1)
X_test = X_test.drop('arrival_date', axis=1)


# In[63]:


#FEATURE SELECTION


# In[64]:


model = RandomForestClassifier(n_estimators=10,random_state=20)


# In[ ]:


rfecv = RFECV(estimator=model, step=1, cv=3, scoring='f1')
selector = rfecv.fit(X_train,y_train)


# In[ ]:



rank=selector.ranking_
feature_rank=pd.DataFrame({'Feature':X_train.columns,'Rank':rank})
feature_selected=(feature_rank[feature_rank['Rank']==1]['Feature']).to_list()
print(feature_selected)
print("The number of features selected",len(feature_selected))


# In[ ]:


#Dropping non-selected features variables


# In[ ]:


X_train=X_train[feature_selected]
X_test=X_test[feature_selected]


# In[ ]:


X_test.shape, y_test.shape


# In[ ]:


def modelEvaluation(modelUsed,Xtrain,ytrain,Xtest,ytest,threshold):
    if 'CatBoostClassifier' in str(modelUsed):
      modelUsed.fit(Xtrain,ytrain,eval_set=(Xtest,ytest),verbose=False,plot=True)
    else:
      modelUsed.fit(Xtrain,ytrain)
    yTrainProbability=modelUsed.predict_proba(Xtrain)[:,1]
    yTrainPrediction=np.where(yTrainProbability>threshold,1,0)
    
    yTestProbability=modelUsed.predict_proba(Xtest)[:,1]
    yTestPrediction=np.where(yTestProbability>threshold,1,0)
    
    print(modelUsed)

    #run only for the best model to save predicted values - if needed
    #We used this to calculate the expected true demand
    yTestPredictionseries = pd.DataFrame(yTestPrediction)
    yTestPredictionseries.to_csv('Predicted_set.csv') 
    files.download('Predicted_set.csv')

    #results
    #confusion matrix
    print("\n")
    print(' Confusion Matrix: Train: \n ')
    data = {'y_Actual': ytrain,'y_Predicted': yTrainPrediction}
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True,cmap='Blues',fmt="d")
    plt.show()
    print("\n")
    print(' Confusion Matrix: Test: \n ')
    data = {'y_Actual': ytest,'y_Predicted': yTestPrediction}
    df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True,cmap='Blues',fmt="d")
    plt.show()
    #auc roc
    print("\n")
    print(' AUC_ROC: Train: \n ', roc_auc_score(ytrain,yTrainProbability))
    print(' AUC_ROC: Test: \n ', roc_auc_score(ytest,yTestProbability))
    # recall score
    print("\n")
    print('Classification Report: Train \n', classification_report(ytrain,yTrainPrediction))
    print('Classification Report: Test \n', classification_report(ytest,yTestPrediction))


# In[ ]:


lr=LogisticRegression(random_state=20)
result_lr=modelEvaluation(lr,X_train,y_train,X_test,y_test,threshold = 0.4)


# In[ ]:



print("The intercept is",lr.intercept_)

weights_f = Series(lr.coef_[0],
                 index=X_train.columns.values)
weights_f2 = DataFrame(dict(weights = weights_f, weights_abs = weights_f.abs()))
weights_f2 = weights_f2.sort_values(by='weights_abs',ascending=False)
weights_f2 = weights_f2.reset_index().rename(columns={'index': 'Features'})

# plot the feature weights
fig = plt.figure()
ax = plt.subplot(111)
ax.barh(weights_f2['Features'][:20], weights_f2['weights'][:20], color='teal')
ax.set_xlabel('feature coefficient weights')


# In[ ]:


dtc=DecisionTreeClassifier(max_depth=10)
modelEvaluation(dtc,X_train,y_train,X_test,y_test,threshold=0.4)


# In[ ]:




