import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
from pandas.io.parsers.readers import read_csv

#Importing
train = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv').dropna().reset_index(drop=True)
#train.info()

# Choose output variable
X = train.drop(columns=['Lead'])
y = train['Lead']

# Split parameters
n_fold = 10
cv = skl_ms.KFold(n_splits = n_fold, random_state = 1, shuffle = True)

# Visualizing
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg') # Output as svg. Else you can try png
from IPython.core.pylabtools import figsize
figsize(10, 6) # Width and hight
np.set_printoptions(precision=3)

# Model
np.random.seed(1)
model = skl_lm.LogisticRegression(solver='liblinear')
X_train, X_test, y_train, y_test = skl_ms.train_test_split(X,y,test_size=0.23)

model.fit(X_train, y_train)
print(model)
predict_prob=model.predict_proba(X_test)
print(model.classes_)
predict_prob[0:5]
prediction=np.empty(len(X_test), dtype=object)
prediction=np.where(predict_prob[:, 0]>=0.5, 'Female', 'Male')
prediction[0:5]

#Confusion matrix
print(pd.crosstab(prediction, y_test), '\n')
print(f"Accuracy: {np.mean(prediction == y_test):.3f}")