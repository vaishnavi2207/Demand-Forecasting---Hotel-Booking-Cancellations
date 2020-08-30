#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



hotel_df = pd.read_csv('C:/Users/vaish/OneDrive/Desktop/PYTHON PROJECTS/hotel_bookings.csv')
hotel_df.head()


# In[ ]:


hotel_df.shape


# In[ ]:


# Data Preprocessing


# In[ ]:


#Missing Values
# % of missing values
perc_null=((hotel_df.isnull().sum()/hotel_df.shape[0])*100).sort_values(ascending=False)
perc_null


# In[ ]:


#Agent and Company are ID columns and have a lot of missing values. 
#So, let's drop them.

hotel_df =  hotel_df.drop(['agent','company'],axis = 1)


# In[ ]:


plt.figure(figsize=(16, 9)) 
corr_matrix = hotel_df.corr()
sns.heatmap(corr_matrix)
plt.show()


# In[ ]:



#Which numerical features are correlated with the label?
corr_with_cancel = corr_matrix["is_canceled"]
corr_with_cancel.abs().sort_values(ascending=False)[1:]


# In[ ]:


#MISSING VALUE IMPUTATION IN TRAIN AND TEST
#Before Imputing missing values, splitting into Train and test to avoid data leakage.


# In[ ]:


X_train, X_test = train_test_split(hotel_df, test_size = 0.30, random_state = 42)
X_train.shape, X_test.shape


# In[ ]:


# needs to be done for test and train separately
def imputation(df):
  df["country"]=df["country"].fillna("Missing")
  df["children"]=df["children"].fillna(df["children"].mode()[0])
  return(df)


# In[ ]:



hotel_df=pd.concat([X_train,X_test])
hotel_df.shape


# In[ ]:


#Feature Engineering


# In[ ]:


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


# In[ ]:


def feature_eng(data):
    data["arrival_date"]=data.apply(arrival_date,axis=1)
    data["arrival_date_weekday"]=data["arrival_date"].dt.weekday
    data["family_or_not"]=data.apply(family_or_not,axis=1)
    data["room_type_changes"]=data.apply(room_type_not_given,axis=1)

    data["Total_No_Of_Nights_Stayed"]=data["stays_in_weekend_nights"]+data["stays_in_week_nights"]
    data["Non_Refund_Flag"]=data["deposit_type"].apply(lambda x:1 if x=="Non Refund" else 0)
    return(data)


# In[ ]:



hotel_df=feature_eng(hotel_df)


# In[ ]:



# Dropping variables used to create new features and reservation_status as label depends on it
hotel_df = hotel_df.drop(['arrival_date_month','arrival_date_year','arrival_date_day_of_month',
                          'reservation_status','reservation_status_date'],axis= 1)


# In[ ]:


#Split into X and Y


# In[ ]:



y=hotel_df["is_canceled"]
X=hotel_df.drop("is_canceled",axis=1)


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42,stratify = y)


# In[ ]:



#y train - Imbalanced data set
y_train.value_counts()/y_train.shape[0]*100


# In[ ]:



#y test
y_test.value_counts()/y_test.shape[0]*100


# In[ ]:


#Balancing the data


# In[ ]:



merged_xy=X_train.merge(y_train,left_index=True,right_index=True)
not_cancel=merged_xy[merged_xy['is_canceled']==0]
cancel=merged_xy[merged_xy['is_canceled']==1]
print(not_cancel.shape)
print(cancel.shape)


# In[ ]:



not_cancel_sampled = resample(not_cancel,
                                replace = False, 
                                n_samples = len(cancel), 
                                random_state = 20)


# In[ ]:


X_train = pd.concat([not_cancel_sampled, cancel])
X_train.shape


# In[ ]:


y_train=X_train['is_canceled']
X_train = X_train.drop('is_canceled',axis = 1)


# In[ ]:



X_train.shape,y_train.shape


# In[ ]:


corr = X_train.corr()
upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
corr_threshold=0.90
to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
to_drop


# In[ ]:


X_train = X_train.drop(X_train[to_drop],axis=1)
X_test = X_test.drop(X_test[to_drop], axis=1)


# In[ ]:


X_train = X_train.drop('arrival_date',axis=1)
X_test = X_test.drop('arrival_date', axis=1)


# In[ ]:


# will ask to save file - run only if needed
#from google.colab import files
#X_test.to_csv('Xtest.csv') 
#files.download('Xtest.csv')


# In[ ]:


model = RandomForestClassifier(n_estimators=10,random_state=20)


# In[ ]:


#MODEL


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





# In[ ]:





# In[ ]:




