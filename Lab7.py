
'''
Lab 7
'''
print ("Lab 7")

import numpy as np
import pandas as pd

##########Part 0 ###########
'''
    1)  load iris-data-1.csv using Pandas and save it as a dataframe (data)
    preprocess your data: 
        a) remove repeated samples (Hint: check "ID")
        b) handle missing data in "color" and preprocess the values for this feature
'''
# YOUR CODE GOES HERE
with open("./data/iris-data-color.csv") as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')
print(data)

##########Part 1 ###########
'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width, color
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''
# YOUR CODE GOES HERE



'''
    2)  what is the error in previous part? how you can fix it?
    fix the error and train your model.
    Hint: pandas.get_dummies()
'''
# YOUR CODE GOES HERE

'''
    3)  test your trained model and evaluate the result for given test set.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE

'''
    4)  use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE


'''
    5)  use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2 and Q3
'''
# YOUR CODE GOES HERE

##########Part 2 ###########

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

