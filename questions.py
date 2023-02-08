from turtle import clear
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from IPython.core.pylabtools import figsize
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score
figsize(10,6) # width and height

train = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv')

train.info()
train.head()
y_train_male = train['Number of male actors']
y_train_female = train['Number of female actors']
x_train = train['Year']
plt.bar(x_train,y_train_male, color='red',label ='Number of male actors')
plt.bar(x_train,y_train_female, label= 'Number of female actors')
plt.xlabel('Year')
plt.ylabel('Number of Actors')
plt.legend()
plt.title('Number of actors in speaking roles')
plt.show

male_sum = y_train_male.sum()
female_sum = y_train_female.sum()
plt.bar(1,y_train_male, color='red',label ='Number of male actors')
plt.bar(1,y_train_female, label= 'Number of female actors')
plt.ylabel('Number of Actors')
plt.legend()
plt.title('Number of actors in speaking roles')
print(plt.show())
male_sum/(male_sum+female_sum)

#train.info()
most_words = np.where(train['Number words male']>train['Number words female'],'Male','Female')
gross_male = 0
gross_female = 0
gross = train['Gross']
for x in range(len(most_words)):
  if most_words[x] == 'Male':
    gross_male += gross[x]
  else:
    gross_female += gross[x]
print('Male:', gross_male/(most_words == 'Male').sum())
print('Female:', gross_female/(most_words == 'Female').sum())
print((gross_male/(most_words == 'Male').sum())/(gross_female/(most_words == 'Female').sum()))
print((most_words == 'Male').sum())
print(len(most_words))
print((most_words == 'Male').sum()/len(most_words))