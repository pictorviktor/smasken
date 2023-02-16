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

#LDA, QDA
train = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv').dropna().reset_index(drop=True)
# train.info()

X = train.drop(columns=['Lead'])
y = train['Lead']


# X_dropped = train.drop(columns=['Lead']).drop(drop_index)
X_dropped = train.drop(columns=['Lead', 'Total words'])
y_dropped = train['Lead']
np.random.seed(1)
K = np.arange(1,200)
miss = []
X_train, X_val, y_train, y_val = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.3)
model = skl_da.QuadraticDiscriminantAnalysis()
model.fit(X_train,y_train)

predict_prob= model.predict_proba(X_val)
print('The class order in the model:')

print(model.classes_)
print(predict_prob[0:5])


prediction = np.empty(len(X_val), dtype=object)
prediction=np.where(predict_prob[0:,0]>= 0.5, 'Female', 'Male')
print('First five prediction:')
print(prediction[0:5], '\n')



#Confusion matrix

print(pd.crosstab(prediction, y_val),'\n')

#Accuracy
print(f"Accuracy:{np.mean(prediction == y_val):.3f}")