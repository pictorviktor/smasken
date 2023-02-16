import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import sklearn.preprocessing as skl_pre
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_me
import matplotlib.pyplot as plt
from pandas.io.parsers.readers import read_csv

# To get nicer plots
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg') # Output as svg. Else you can try png
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
np.set_printoptions(precision=3)

"""initial manipulation of data

Logistic regression
"""

train_data = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv').dropna().reset_index(drop=True)
# train.info()

#X = train.drop(columns=['Lead'])
#y = train['Lead']

n_fold = 10
cv = skl_ms.KFold(n_splits = n_fold, random_state = 1, shuffle = True)

np.random.seed(1)

#X_train, X_val, y_train, y_val = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.3)

trainI=np.random.choice(train_data.shape[0], size=800, replace=False)
trainIndex=train_data.index.isin(trainI)
train = train_data.iloc[trainIndex]
test = train_data.iloc[~trainIndex]

model = skl_lm.LogisticRegression(solver='liblinear')
X_train= train.drop(columns=['Lead', 'Total words'])
Y_train=train['Lead']
X_test=test.drop(columns=['Lead', 'Total words'])
Y_test=test['Lead']

model.fit(X_train,Y_train)

print('Model summary:')
print(model)

predict_prob=model.predict_proba(X_test)
print('The class order in the model:')
print(model.classes_)
print('Examples of predicted probabilities for the above classes:')
predict_prob[0:5]

prediction=np.empty(len(X_test), dtype=object)
prediction=np.where(predict_prob[:,0]>=0.5, 'Female', 'Male')
prediction[0:5]

#Confusion matrix
print('Condusion matrix: \n')
print(pd.crosstab(prediction, Y_test), '\n')

#Accuracy 
print(f"Accuracy: {np.mean(prediction== Y_test): .3f}")

"""LDA and QDA 





"""

np.random.seed(1)

trainI=np.random.choice(train_data.shape[0], size=850, replace=False)
trainIndex=train_data.index.isin(trainI)
train = train_data.iloc[trainIndex]
test = train_data.iloc[~trainIndex]

model=skl_da.LinearDiscriminantAnalysis()

X_train= train.drop(columns=['Lead', 'Total words'])
Y_train=train['Lead']
X_test=test.drop(columns=['Lead', 'Total words'])
Y_test=test['Lead']

model.fit(X_train, Y_train)

predict_prob=model.predict_proba(X_test)
print('The class order in the model:')

print(model.classes_)
print('Examples of predicted probabilities for the above classes:')
with np.printoptions(suppress=True, precision=3):
  print(predict_prob[0:10])

prediction=np.empty(len(X_test), dtype=object)
prediction=np.where(predict_prob[:,0]>=0.5, 'Female', 'Male')
print('First five predictions:')
print(prediction[0:5],'\n')


#Confusion matrix
print('Confusion matrix')
print(pd.crosstab(prediction, Y_test), '\n')

#Accuracy 
print(f"Accuracy: {np.mean(prediction==Y_test): .3f}")