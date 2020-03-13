
'''
Lab 5
'''
print ("Lab 5")

import numpy as np

##########Part 0 ###########

'''
    1)  load iris dataset using Pandas and save it as a dataframe (data)
    print out the columns names
        
'''
import pandas as pd

with open("./data/iris-data-color.csv") as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')

print(data.columns)
##########Part 1 ###########
'''
    1) using pandas.DataFrame.drop_duplicates to drop duplications in "ID" (keep the first instance) and save the resulted data frame in a new datafram (df). print out the shape of df.

'''
# YOUR CODE GOES HERE
print("\nlength before: "+ str(len(data)) )
data = data.drop_duplicates("ID", keep='first')
print("length after: "+ str(len(data)) )

'''
    2)  plot the bar graph for df['color']
    Hint: pandas.DataFrame.plot.bar()
    
'''
import matplotlib.pyplot as plt
# YOUR CODE GOES HERE
#print(data["color"].value_counts())
data["color"].value_counts().plot.bar()
#plt.show()


'''
    3)  how many missing data ponits we have in 'color'?
    Hint: pandas.DataFrame.dropna()
    
'''
# YOUR CODE GOES HERE

drop = data.dropna(how='any', subset=['color'])
print( " # of missing points in color " + str(len(data) - len(drop)) + '\n')



'''
    2)  Repeat Q2, Do you see any difference? Why?
    
'''
# YOUR CODE GOES HERE
drop["color"].value_counts().plot.bar()
#plt.show()


'''
    5)  make the values in 'color' to be consistant and remove the unkown values
    Hint: pandas.DataFrame.replace()
       
'''
# YOUR CODE GOES HERE

# remove: None, none, U, Nan
# add to blue: Blue, Bluee

drop = drop.replace(['Blue', 'Bluee'],'blue')
drop = drop.replace(['Red'],'red')
drop = drop.replace(['None','none', 'U', 'Nan'])
drop["color"].value_counts().plot.bar()
# plt.show()


##########Part 2 ###########

'''
    1) repeat Q1 in part 1

'''
# YOUR CODE GOES HERE
with open("./data/iris-data-color.csv") as csvfile:
    data2 = pd.read_csv(csvfile, delimiter=',')



'''
    2)  replace missing value with 'pink' 
    Hint: pandas.DataFrame.fillna()
    
'''
# YOUR CODE GOES HERE
data2 = data2.fillna('pink')



##########Part 3 ###########
'''
    1)  train a KKN classifier for your data (K = 5). Features: sepal_length,sepal_width,petal_length,petal_width
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''
# YOUR CODE GOES HERE 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
data2 = data2.drop(columns=['ID','color'])
knn.fit(data2[["sepal_length","sepal_width","petal_length","petal_width"]],data2["species"])


'''
    2)  test your classifier and report the performance 
'''
X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE
from sklearn.metrics import classification_report
Y_True = Y_test
Y_Pred = knn.predict(X_test)

print("Predictions:"+str(Y_Pred))
print("True"+str(Y_True)+"\n")

report = classification_report(Y_True, Y_Pred)
print( report )


'''
    3)  repeat q1 and q2 for "manhattan" distance function
'''
# YOUR CODE GOES HERE TODO:
knnManhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knnManhattan.fit(data2[["sepal_length","sepal_width","petal_length","petal_width"]],data2["species"])
Y_True = Y_test
Y_Pred = knnManhattan.predict(X_test)
report = classification_report(Y_True, Y_Pred)
print("\n\nManhattan:\n"+ str(report) )

##########Part 4 ###########
'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''
# YOUR CODE GOES HERE TODO:
knnSeven = KNeighborsClassifier(n_neighbors=11)
knnSeven.fit(data2[["sepal_length","sepal_width","petal_length","petal_width"]],data2["species"])

'''
    2)  test your classifier and report the performance 
'''
# YOUR CODE GOES HERE TODO:
X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])

Y_True = Y_test
Y_Pred = knnManhattan.predict(X_test)
report = classification_report(Y_True, Y_Pred)
print("\n K = 7:\n"+ str(report) )

'''
    3)  repeat q1 and q2 for "manhattan" distance function
'''
# YOUR CODE GOES HERE TODO:
knnManhattan = KNeighborsClassifier(n_neighbors=11, metric='manhattan')
knnManhattan.fit(data2[["sepal_length","sepal_width","petal_length","petal_width"]],data2["species"])
Y_True = Y_test
Y_Pred = knnManhattan.predict(X_test)
report = classification_report(Y_True, Y_Pred)
print("\n\nK = 7 Manhattan:\n"+ str(report) )


##########Part 5 ########### optional.....

'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width, color
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''

# YOUR CODE GOES HERE

'''
    2)  what is the error in previous part? how you can fix it?
    Hint: pandas.get_dummies()
'''


##########Part 6 ###########

'''
    1)  Features: sepal_length,sepal_width,petal_length,petal_width.
    use cross validation (CV: Hold-out validation) to find the best value for k.  (try k = 1, 3, 5, 7, 9, 11)
    Hint: use train_test_split from sklearn.model_selection (use 30% of your data as a test set)
    plot the results and pick the best value for k
'''

# YOUR CODE GOES HERE

'''
    2)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use KFold from sklearn.model_selection
    plot the results and pick the best value for k
'''

# YOUR CODE GOES HERE

'''
    3)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use cross_val_score from sklearn.model_selection
    plot the results and pick the best value for k
'''
# YOUR CODE GOES HERE


##########Part 7 ###########
'''
    1)  what are other ways to implement 5-fold CV in sklearn?
'''