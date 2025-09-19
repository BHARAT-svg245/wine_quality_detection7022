import os
os.environ["LOKY_MAX_CPU_COUNT"]="4"
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import KFold,StratifiedShuffleSplit,train_test_split,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#Reading of DataFrame
p=pd.read_csv("WineQT.csv")
df=pd.DataFrame(p)
df=df.drop(["Id"],axis=1)
#Droping the lower class value
m1=np.where(df["quality"]==3)[0] 
df=df.drop(m1,axis=0).reset_index(drop=True)
m2=np.where(df["quality"]==8)[0]
df=df.drop(m2,axis=0).reset_index(drop=True)
m3=np.where(df["quality"]==4)[0]
df=df.drop(m3,axis=0).reset_index(drop=True)
#print(df.describe())
splt=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_ind,test_ind in splt.split(df,df["quality"]):
    str_train_set=df.loc[train_ind]
    str_test_set=df.loc[test_ind]
# #Training Data
train_feature=str_train_set.drop("quality",axis=1)
train_label=str_train_set["quality"]
print(train_feature)
new_train_feature=train_feature
new_train_label=train_label
#making the pipeline
#my_pipe=Pipeline(steps=[("scaler",StandardScaler()),("classifier",GradientBoostingClassifier())])
# my_pipe=Pipeline(steps=[("scaler",StandardScaler()),("classifier",AdaBoostClassifier())])
my_pipe=Pipeline(steps=[("scaler",StandardScaler()),("classifier",RandomForestClassifier())])
cvs=cross_val_score(my_pipe,new_train_feature,new_train_label,cv=5)
#print(cvs)
#Hyperparameter tuning by grid_search
#param_grid={"classifier__n_estimators":[50,100,200],"classifier__max_depth":[None,10,20],"classifier__min_samples_split":[1,3,5],"classifier__min_samples_leaf":[1,2]}
# grid_search=GridSearchCV(my_pipe,param_grid=param_grid,cv=5,scoring="accuracy")
# grid_search.fit(new_train_feature,new_train_label)
# print("best_param:",grid_search.best_params_)
# print("best_score:",grid_search.best_score_)
#fitting the data into pipe
my_pipe.fit(new_train_feature,new_train_label)
#Testing Data
test_feature=str_test_set.drop("quality",axis=1)
new_test_feature=(test_feature)
test_label=str_test_set["quality"]
#making the prediction
test_pred=my_pipe.predict(new_test_feature)
train_pred=my_pipe.predict(new_train_feature)
#Accuracy of the model
test_accuracy=accuracy_score(test_pred,test_label)
#Evaluating the bias-variance score
training_accuracy=accuracy_score(new_train_label,train_pred)
#Saving the model for use
#import joblib
#joblib.dump(my_pipe,"Wine_quality_prediction.joblib")
# loaded_model=joblib.load("Wine_qualti_predictio.joblib")
# f=np.array([l1])
# y_pred=loaded_model.predict(f)
# print(y_pred)


#print(test_accuracy)
