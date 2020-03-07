
'''
Lab 6
'''
print ("Lab 6")
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

with open("./data/iris-data-color.csv") as csvfile:
    modifiedData = pd.read_csv(csvfile, delimiter=',')

modifiedData = modifiedData.drop_duplicates("ID", keep = 'first')
modifiedData = modifiedData.replace(['Blue', 'Bluee'],'blue')
modifiedData = modifiedData.replace("Red", "red")
print(modifiedData)
modifiedData = modifiedData.replace(['None', 'none', 'U', 'Nan', 'NaN'], float('nan'))
modifiedData.dropna(how='any', inplace=True)
modifiedData = modifiedData.reset_index(drop=True)
print(modifiedData)


##########Part 1 ###########
'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width, color
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''

# YOUR CODE GOES HERE
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 11)
modifiedData0 = modifiedData.drop(columns = ["ID"])

'''
    2)  what is the error in previous part? how you can fix it?
    fix the error and train your model.
    Hint: pandas.get_dummies()
'''
# YOUR CODE GOES HERE
colors = pd.get_dummies(modifiedData0['color'])
modifiedData0 = pd.concat([modifiedData0, colors], axis=1)
# modifiedData0 = modifiedData0.drop(columns = ["color"])
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(modifiedData0[["sepal_length","sepal_width","petal_length","petal_width","blue","pink","purple","red"]], modifiedData0["species"])
# print(modifiedData0.head(10))
'''
    3)  test your trained model and evaluate the result for given test set.
'''
X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE

colors = ["blue", "pink", "purple", "red"]
colorDataFrame = pd.DataFrame(colors, columns=['color']) 

def transform(array, colors):
    getColors = pd.get_dummies(colors['color'])
    print(getColors)
    for i in range(len(array)):
        for c in range(len(colors)):

            # if array[i][4] == colors[c]:
            #     array[i].pop()
            #     array[i].append(getColors[c])
    return array

    # print(array[0])
    # remove end & append color
    # data = pd.concat([data, colors], axis=1)
    
a = transform(X_test, colorDataFrame)

# from sklearn.metrics import classification_report
# Y_True = Y_test
# Y_Pred = knn.predict(X_test)

# # print("Predictions:"+str(Y_Pred))
# # print("True"+str(Y_True)+"\n") 

# report = classification_report(Y_True, Y_Pred)
# print( report )



'''
    4)  use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE
# from sklearn.feature_extraction import DictVectorizer

# l = []
# for i in range(len(modifiedData['color'])):
#     l.append( { (i) , (modifiedData['color'][i]) } )
# leng = len(modifiedData['color'])-1
# print(l[leng])
# print(l)


# dataDic = modifiedData.to_dict()
# print(dataDic)

# v = DictVectorizer()
# colors = v.fit_transform(dataDic)

# print(colors)

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
    1)  what are other ways to implement 5-fold CV in sklearn?
'''
