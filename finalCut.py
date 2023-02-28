# Grupp 26

# Dependencies
import numpy as np
import pandas as pd
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# fetch data for intial questions
trainInt = pd.read_csv('/Users/admin/Documents/CodeProjects/numpy_projects/train.csv')

# To get nicer plots
from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg') # Output as svg. Else you can try png
# figsize(10, 6) # Width and hight
# np.set_printoptions(precision=3)

# Has gender balance in speaking roles changed over time (i.e. years)?
def balance_plot(trainInt):
    y_train_male = trainInt['Number of male actors']
    y_train_female = trainInt['Number of female actors']
    x_train = trainInt['Year']
    plt.bar(x_train,y_train_male, color='red',label ='Number of male actors')
    plt.bar(x_train,y_train_female, label= 'Number of female actors')
    plt.xlabel('Year')
    plt.ylabel('Number of Actors')
    plt.legend()
    plt.title('Number of actors in speaking roles')
    plt.show

# Do men or women dominate speaking roles in Hollywood movies?
def men_vs_women(trainInt):
    y_train_male = trainInt['Number of male actors']
    y_train_female = trainInt['Number of female actors']
    male_sum = y_train_male.sum()
    female_sum = y_train_female.sum()
    #plt.bar(1,y_train_male, color='red',label ='Number of male actors')
    #plt.bar(1,y_train_female, label= 'Number of female actors')
    # plt.ylabel('Number of Actors')
    # plt.legend()
    # plt.title('Number of actors in speaking roles')
    # plt.show
    return male_sum/(male_sum+female_sum)

# Do films in which men do more speaking make a lot more money than films in which women speak more?

def money_maker(trainInt):
    #train.info()
    most_words = np.where(trainInt['Number words male']>trainInt['Number words female'],'Male','Female')
    gross_male = 0
    gross_female = 0
    gross = trainInt['Gross']
    for x in range(len(most_words)):
        if most_words[x] == 'Male':
            gross_male += gross[x]
        else:
            gross_female += gross[x]

    print('Male:', gross_male/(most_words == 'Male').sum())
    print('Female:', gross_female/(most_words == 'Female').sum())
    print((gross_male/(most_words == 'Male').sum())/(gross_female/(most_words == 'Female').sum()))
    Actor = ['Male','Female']
    Gross = [(gross_male/(most_words == 'Male').sum()),(gross_female/(most_words == 'Female').sum())]

    plt.rcParams["figure.autolayout"] = True
    plt.bar(Actor,Gross)
    plt.grid('on')
    plt.ylabel('Averge Gross')
    plt.title('Average Gross of movies with most spoken words')

# Initial manipulation of data

train = pd.read_csv('train.csv').dropna().reset_index(drop=True)

X = train.drop(columns=['Lead'])
y = train['Lead']

#n_fold = 10
#cv = skl_ms.KFold(n_splits = n_fold, random_state = 1, shuffle = True)

#Trying PCA
def PCA(X):
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
    X_pca = X.dot(projection_matrix)

    ### Dropping the outliers
    drop_index=[]
    for x in X_pca.T:
        if X_pca.T[x][0]>30000 and X_pca.T[x][1]>10000:
            drop_index.append(x)
        elif X_pca.T[x][0]>60000:
            drop_index.append(x) 
        print('drop:',drop_index)
        X_pca_dropped = X_pca.drop(drop_index)

    fig = plt.figure()
    ax1 = plt.axes(projection ="3d")
    ax1.scatter3D(X_pca[0],X_pca[1],X_pca[2])
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.set_zlabel('Component 3')

    fig2 = plt.figure()
    ax2 = plt.axes(projection ="3d")
    ax2.scatter3D(X_pca_dropped[0],X_pca_dropped[1],X_pca_dropped[2])
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')


# Trying variance of attributes
def att_variance(X):
    data = X
    data = data - data.mean()
    print(data.var())
    variance = data.var()
    columns = data.columns
    variable = [ ]

    for i in range(0,len(variance)):
        if variance[i]<=100: 
            variable.append(columns[i])

    print(variable)

### The final data output
X_dropped = train.drop(columns=['Lead', 'Total words'])
#X_dropped = train.drop(columns=['Lead','Number of male actors', 'Number of female actors', 'Mean Age Male', 'Mean Age Female']).drop(drop_index)
y_dropped = train['Lead']

#Logistic Regression
def log_reg(X_dropped, y_dropped):
    np.random.seed(1)
    X_train, X_test, Y_train, Y_test = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.3)

    model = skl_lm.LogisticRegression(solver='liblinear')
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
    print('Confusion matrix: \n')
    print(pd.crosstab(prediction, Y_test), '\n')

    #Accuracy 
    print(f"Accuracy: {np.mean(prediction== Y_test): .3f}")

    scores = skl_ms.cross_val_score(model, X_dropped, y_dropped, cv = 10)
    mean_accuracy = np.mean(scores)
    print(scores)
    print('mean accuracy', mean_accuracy)

# LDA, QDA
def LDA_QDA(X_dropped, y_dropped):
    np.random.seed(1)

    X_train, X_test, Y_train, Y_test = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.3)

    model=skl_da.QuadraticDiscriminantAnalysis()
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

    scores = skl_ms.cross_val_score(model, X_dropped, y_dropped, cv = 10)
    mean_accuracy = np.mean(scores)
    print(scores)
    print('mean accuracy', mean_accuracy)

# kNN
def kNN(X_dropped,y_dropped):
    K = np.arange(1,200)
    miss = []
    X_train, X_val, y_train, y_val = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.3)

    for k in K:
        model = skl_nb.KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        miss.append(np.mean(prediction != y_val))

    plt.plot(K, miss)
    plt.title('Missclassification of error on validation set for kNN')
    plt.xlabel('Number of neighbors $k$')
    plt.ylabel('Missclassification error')
    plt.grid(True)
    plt.show()
    print(K[np.argmin(miss)])


    interestingK = [5,7,13,15]
    for k in interestingK:
        np.random.seed(1)
        model = skl_nb.KNeighborsClassifier(n_neighbors = k)
        model.fit(X_dropped, y_dropped)

        scores = skl_ms.cross_val_score(model, X_dropped, y_dropped, cv = 10)
        mean_accuracy = np.mean(scores)
        print('k:',k,'score:', scores)
        print('mean accuracy:', mean_accuracy)

### Tree-based methods

def random_forest(X_dropped, y_dropped):
    # setting up the training/testing-data
    np.random.seed(4)
    X_train, X_val, y_train, y_val = skl_ms.train_test_split(X_dropped,y_dropped,test_size=0.25) 
    print(train.info())

    # Modeling
    model = RandomForestClassifier(n_estimators=200) #Adding oob_score, max_depth or more estimators- 
    #does not result in better model in this case
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    error = round(np.mean(model.predict(X_val) != y_val),3)
    print('Test error:', error,' and accuracy', accuracy)

    # Perform cross-validation
    scores = skl_ms.cross_val_score(model, X_dropped, y_dropped, cv = 10)
    mean_accuracy = np.mean(scores)
    print(scores)
    print('mean accuracy', mean_accuracy)

    # Calculate the mean accuracy across the 5 folds
    # mean_accuracy = np.mean(scores)
    # print('scores',scores)
    # print('Cross validation mean accuracy', mean_accuracy)

#Running models
balance_plot(trainInt)
# men_vs_women(trainInt)
# money_maker(trainInt)
# kNN(X_dropped, y_dropped)
# LDA_QDA(X_dropped, y_dropped)
# log_reg(X_dropped, y_dropped)
# random_forest(X_dropped, y_dropped)