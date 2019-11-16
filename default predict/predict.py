# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostClassifier
# load test data
def load_test_fs(filename):
    default = pd.read_csv(filename, index_col = "Test_ID")
    #default = pd.read_csv(filename, index_col = "Test_ID")
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
    pay_bill_features = ['pay_amt1','pay_amt2','pay_amt3','pay_amt4','pay_amt5','pay_amt6','bill_amt1','bill_amt2','bill_amt3','bill_amt4','bill_amt5','bill_amt6']
    for p in pay_bill_features:
        default[p] = default[p]/default['limit_bal']
    default.rename(columns = {'y':'default'}, inplace = True)
    return default

def output_preds(preds,name):
    result=preds.iloc[:,0:1]
    result.to_csv(name+'.csv', encoding='utf-8', index=False)
    return

    

  
# extract features from test data
def test_type(test_fs):
    x_Test=test_fs
    robust_scaler= RobustScaler()
    x_Test= robust_scaler.fit_transform(x_Test)
    return x_Test

def test_id(test_fs,filename):
    #default= pd.read_csv(str(sys.argv[2]))
    default= pd.read_csv(filename)
    x_Test = default['Test_ID']
    return x_Test


def Cat_Boost(x_test,x_id):
    catboost=CatBoostClassifier()
    model=catboost.load_model("my_model")
    y_pred_prob=model.predict_proba(x_test)[:,1]
    dic={
            "Rank_ID":x_id,
            "prob1":y_pred_prob
            }
    return  pd.DataFrame(dic).sort_values(by=['prob1'],ascending=False)


# the main function
if __name__ == '__main__':
    ptest_fs= load_test_fs(str(sys.argv[2]))
    ptest_x = test_type(ptest_fs)
    ptestid= test_id(ptest_fs,str(sys.argv[2]))
    output_preds(Cat_Boost(ptest_x,ptestid),'public')
    vtest_fs= load_test_fs(str(sys.argv[3]))
    vptest_x = test_type(vtest_fs)
    vptestid= test_id(vtest_fs,str(sys.argv[3]))
    output_preds(Cat_Boost(vptest_x,vptestid),'private')

