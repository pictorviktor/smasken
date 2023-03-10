from turtle import clear
import numpy as np
import pandas as pd
from IPython.core.pylabtools import figsize
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from subprocess import call

figsize(10,6) # width and height

# Fetch data from csv
train = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv')
#setting up the training/testing-data
np.random.seed(4)
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
# print('Naive guess is', number_of_data_points, 'out of', train.shape[0], 'is male')
# print('this equals to', round(number_of_data_points/train.shape[0], 3), 'of the whole dataset')



# #Visualizing the graph
# dot_data = export_graphviz(
# model.estimators_[99], 
# out_file=None, 
# feature_names= x_train.columns, 
# class_names = model.classes_, 
# filled=True,
# impurity=True, 
# rounded=True, 
# leaves_parallel=True, 
# proportion=True
# )
# graph = graphviz.Source(dot_data)
# print(graph)





