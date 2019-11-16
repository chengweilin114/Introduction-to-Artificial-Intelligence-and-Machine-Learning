# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 23:10:35 2017

@author: 羅弘
"""
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier
# load train data
def load_train_fs():
    default = pd.read_csv(str(sys.argv[1]), index_col = "Train_ID")
    #default = pd.read_csv("Train.csv", index_col = "Train_ID")
    default.rename(columns=lambda x:x.lower(),inplace=True)
    default['grad_school'] = (default['education'] == 1).astype('int')
    default['university'] = (default['education'] == 2).astype('int')
    default['high_school'] = (default['education'] == 3).astype('int')
    default['other_education'] = (default['education'] > 3).astype('int')
    default.drop('education', axis = 1, inplace=True)
    default['male'] = (default['sex'] == 1).astype('int')
    default.drop('sex', axis = 1, inplace = True)
    default['married'] = (default['marriage'] == 1).astype('int')
    default['divorced'] = (default['marriage'] == 3).astype('int')
    default['single'] = (default['marriage'] == 2).astype('int')
    default['other_marriage'] = (default['marriage'] == 0).astype('int')
    default.drop('marriage', axis = 1, inplace = True)
    default['default']=(default['y']==1).astype('int')
    default.drop('y',axis=1,inplace=True)
    pay_bill_feature = ['pay_amt1','pay_amt2','pay_amt3','pay_amt4','pay_amt5','pay_amt6','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6']
    for p in pay_bill_feature:
        default[p] = default[p]/default['limit_bal']
    default.rename(columns = {'y':'default'}, inplace = True)
    return default
#train arrange
def train_type(train_fs,targetname):
    train_x=train_fs.drop(targetname,axis=1)
    robust_scaler= RobustScaler()
    train_x= robust_scaler.fit_transform(train_x)
    train_y= train_fs[targetname]
    return train_x,train_y
#train model produced
def Cat_Boost(x_train,y_train):
    model=CatBoostClassifier(iterations=1000,learning_rate=0.005, depth=12, loss_function='Logloss',verbose=True)
    model.fit(x_train,y_train)
    model.save_model("my_model")
    return 
# the main function
if __name__ == '__main__':
    train_fs = load_train_fs()
    train_x,train_y = train_type(train_fs,'default')
    Cat_Boost(train_x,train_y)