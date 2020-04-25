
'''
Lab 7 and Lab 8 Solution
'''


import numpy as np

##########Part 0 ###########

'''
    1)  load iris-data-1.csv using Pandas and save it as a dataframe (data)
    preprocess your data: 
        a) remove repeated samples (Hint: check "ID")
        b) handle missing data in "color" and preprocess the values for this feature
        
'''
# YOUR CODE GOES HERE
import pandas as pd
with open('iris-data.csv') as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')

df = data.drop_duplicates("ID", keep = 'first')
dff = df.dropna()
dff = dff.replace("Blue", "blue")
dff = dff.replace("Bluee", "blue")
dff = dff.replace("Red", "red")
dff = dff.replace("None", "none")
dff = dff.replace("Nan", "none")
dff = dff.replace("U", "none")
dff = dff[dff.color !='none']
dff = dff.reset_index(drop=True)
##########Part 1 ###########

'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width, color
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''

# YOUR CODE GOES HERE
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 11)

dff = dff.drop(columns = ["ID"])
X = dff[['sepal_length',  'sepal_width',  'petal_length',  'petal_width', 'color']]
y = (dff["species"])
#neigh.fit(X,y)  #error


'''
    2)  what is the error in previous part? how you can fix it?
    fix the error and train your model.
    Hint: pandas.get_dummies()
'''
# YOUR CODE GOES HERE
dummy = pd.get_dummies(X['color'])
X_new = pd.concat([dff[['sepal_length',  'sepal_width',  'petal_length',  'petal_width']],dummy],axis = 1)
neigh.fit(X_new,y)    # to be used for solution2 in the next question


'''
    3)  test your trained model and evaluate the result for given test set.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE
from sklearn.metrics import classification_report

# one solution is to combine test and train and to do the transformation, another solution is to transform the test data manually:
#solution 1
df_x_test = pd.DataFrame(
    {'sepal_length': X_test[:,0],
    'sepal_width': X_test[:,1],
    'petal_length': X_test[:,2],
    'petal_width': X_test[:,3],
    'color': X_test[:,4]})

combined = pd.concat([X, df_x_test], ignore_index=True)
dummy = pd.get_dummies(combined['color'])
combined = combined.drop(columns = 'color') 
combined = pd.concat([combined, dummy], axis='columns')

X_train_new = combined.head(-len(X_test))
X_test_new= combined.tail(len(X_test))

neigh1 = KNeighborsClassifier(n_neighbors=11)
neigh1.fit(X_train_new, y)
pred = neigh1.predict(X_test_new)
print (classification_report(Y_test,pred))

#solution 2
X_test_new = np.asarray([[5 , 1, 0.2 , 5, 0, 0,0,1],[0.9 , 7, 6.2 , 2.1,0,0,0,1], [0.9 , 7, 6.2 , 2.1,0,1,0,0] , [1.9 , 4, 5 , 0.1,0,0,1,0], [5.9 , 3.3, 0.2 , 2.7,1,0,0,0]])
pred = neigh.predict(X_test_new)
print (classification_report(Y_test,pred))

'''
    4)  use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE
from sklearn.feature_extraction import DictVectorizer

myDict = (X[['color']]).to_dict('records')
vec = DictVectorizer()
new_myDict = vec.fit_transform(myDict).toarray()
colors = pd.DataFrame.from_dict(new_myDict)

X1 = X.drop(['color'], axis=1)
X1 = X1.reset_index()

X_train_new = pd.concat([X1, colors], axis=1)

'''
    5)  use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2 and Q3
'''
# YOUR CODE GOES HERE
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories = "auto",sparse = False)
le = preprocessing.LabelEncoder()

X['color']= le.fit_transform(X["color"])
temp = ohe.fit_transform(X[["color"]])
X1 = X.drop(['color'], axis=1)
X1 = X1.reset_index()
ohe_column = pd.DataFrame(temp, columns = ["red","pink",'blue','purple'])
X_train_new = pd.concat([X1,ohe_column],axis = 1)

neigh2 = KNeighborsClassifier(n_neighbors=11)
neigh2.fit(X_train_new, y)

##########Part 2 ###########

'''
    1)  Features: sepal_length,sepal_width,petal_length,petal_width.
    use cross validation (CV: Hold-out validation) to find the best value for k.  (try k = 1, 3, 5, 7, 9, 11)
    Hint: use train_test_split from sklearn.model_selection (use 30% of your data as a test set)
    plot the results and pick the best value for k
'''
import pandas as pd
with open('iris-data.csv') as csvfile:
    dff = pd.read_csv(csvfile, delimiter=',')
X = dff[['sepal_length',  'sepal_width',  'petal_length',  'petal_width']]
y = (dff["species"])    
    
# YOUR CODE GOES HERE
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
result = []
K_list = [1,3,5,7,9,11]
for k in K_list:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    result.append(metrics.accuracy_score(y_test, y_pred))
 

import matplotlib.pyplot as plt
plt.plot(K_list,result)   

'''
    2)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use KFold from sklearn.model_selection
    plot the results and pick the best value for k
'''

# YOUR CODE GOES HERE
from sklearn.model_selection import KFold
from sklearn import metrics
import statistics

kf = KFold(n_splits=5, shuffle=True, random_state = 123)

result = []
K_list = [1,3,5,7,9,11]
for k in K_list:
    result_fold = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        result_fold.append(metrics.accuracy_score(y_test, y_pred))
    result.append(statistics.mean(result_fold))
    
plt.plot(K_list,result) 
    
'''
    3)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use cross_val_score from sklearn.model_selection
    plot the results and pick the best value for k
'''
# YOUR CODE GOES HERE
from sklearn.model_selection import cross_val_score
result = []
K_list = [1,3,5,7,9,11]
for k in K_list:
    neigh = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(neigh, X, y, cv=5, scoring='accuracy')
    result.append(scores.mean())
    
plt.plot(K_list,result)     
##########Part 3 ###########
'''
    1)  Try to normalize your features, and repeat Q2 in part 2
    Hint: use sklearn.preprocessing.scalar(), StandardScaler(),MinMaxScaler(), MaxAbsScaler()
    
'''

##########Part 4 ###########
'''
    1)  train a DT classifier for your data  (criterion = 'entropy'). 
    Features: sepal_length,sepal_width,petal_length,petal_width
    use predict_proba() for testing your trained model 
 
'''

X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])

'''
    2)  train a DT classifier for your data  (criterion = 'Gini'). 
    Features: sepal_length,sepal_width,petal_length,petal_width
    use predict_proba() for testing your trained model 
 
'''

##########Part 5 ###########
'''
    1)  train a NB classifier for your data. 
    Features: sepal_length,sepal_width,petal_length,petal_width.
    Which NB model you have to use here?
 
'''

X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])

