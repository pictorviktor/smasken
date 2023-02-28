import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
from pandas.io.parsers.readers import read_csv
import sklearn.preprocessing as skl_pre
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.metrics as skl_me
import matplotlib.pyplot as plt
from turtle import clear
from IPython.core.pylabtools import figsize
from IPython.display import Image
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
import graphviz
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from subprocess import call
import pydot

### fetch data
train = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv')

# train.info()

X = train.drop(columns=['Lead'])
y = train['Lead']

n_fold = 10
cv = skl_ms.KFold(n_splits = n_fold, random_state = 1, shuffle = True)

#Trying PCA
data = X
data = data - data.mean()
cov_mat = data.cov()
cov_mat
eig_values, eig_vectors = np.linalg.eig(cov_mat)

e_indices = np.argsort(eig_values)[::-1]
eigenvectors_sorted = eig_vectors[:,e_indices]

variance_explained = []
for i in eig_values:
    variance_explained.append((i/sum(eig_values))*100)


#### Showing the PCA
with plt.style.context('ggplot'):
    plt.figure(figsize=(6, 4))
plt.bar(range(13), variance_explained, alpha=0.5, align='center',
            label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend()
plt.tight_layout()

projection_matrix = (eigenvectors_sorted.T[:][:3]).T
# print(projection_matrix)
X_pca = X.dot(projection_matrix)
print(X_pca.head())
# print(X.head())

### Dropping the outliers
drop_index=[]
for x in X_pca.T:
  if X_pca.T[x][0]>30000 and X_pca.T[x][1]>10000:
    drop_index.append(x)
  elif X_pca.T[x][0]>60000:
    drop_index.append(x) 
print('drop:',drop_index)
X_pca_dropped = X_pca.drop(drop_index)


# fig = plt.figure()
# ax1 = plt.axes(projection ="3d")
# ax1.scatter3D(X_pca[0],X_pca[1],X_pca[2])
# ax1.set_xlabel('Component 1')
# ax1.set_ylabel('Component 2')
# ax1.set_zlabel('Component 3')

# fig2 = plt.figure()
# ax2 = plt.axes(projection ="3d")
# ax2.scatter3D(X_pca_dropped[0],X_pca_dropped[1],X_pca_dropped[2])
# ax2.set_xlabel('Component 1')
# ax2.set_ylabel('Component 2')
# ax2.set_zlabel('Component 3')

# fig3 = plt.figure()
# ax3 =plt.axes()
# ax3.scatter(X_pca[0],X_pca[1])

# fig4 = plt.figure()
# # ax = plt.axes(projection ="3d")
# ax4 =plt.axes()
# ax4.scatter(X_pca[0],X_pca[2])


### 
X_dropped = train.drop(columns=['Lead', 'Total words'])#.drop(drop_index)
#X_dropped = train.drop(columns=['Lead','Number of male actors', 'Number of female actors', 'Mean Age Male', 'Mean Age Female']).drop(drop_index)
y_dropped = train['Lead']#.drop(drop_index)

print(X_dropped.info())

### RANDOM FOREST

figsize(10,6) # width and height

#setting up the training/testing-data
np.random.seed(1)
trainingIndex = np.random.choice(train.shape[0], size=780, replace=False)
trainingSet = train.iloc[trainingIndex] 
testSet = train.iloc[~train.index.isin(trainingIndex)]   
print(train.info())
# print(train.head())

x_train = trainingSet.copy().drop(columns=['Lead'])
y_train = trainingSet['Lead']
x_test = testSet.copy().drop(columns=['Lead'])
y_test = testSet['Lead']

# Modeling
model = RandomForestClassifier(n_estimators=200) #Adding oob_score, max_depth or more estimators- 
#does not result in better model in this case
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
error = round(np.mean(model.predict(x_test) != y_test),3)
print('Test error:', error,' and accuracy', accuracy)

# Perform cross-validation
scores = cross_val_score(model, x_train, y_train, cv=10)

# Calculate the mean accuracy across the 5 folds
mean_accuracy = np.mean(scores)
print('scores',scores)
print('Cross validation mean accuracy', mean_accuracy)
#Naive guess

naive_guess = train[train['Lead'] == 'Male']
number_of_data_points = naive_guess.shape[0]

### LOGISTIC REGRESSION

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

### QDA

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