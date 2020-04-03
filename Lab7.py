
'''
Lab 7
'''
print ("Lab 7 ")

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

with open('./data/iris-data-color.csv') as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')
modifiedData = data.drop_duplicates("ID", keep = 'first')

modifiedData = modifiedData.replace(["Blue", "Bluee"], "blue")
modifiedData = modifiedData.replace("Red", "red")
modifiedData = modifiedData.replace(["None", "Nan", "U", "none"], float('nan'))

modifiedData.dropna(inplace = True)
modifiedData = modifiedData.reset_index(drop = True)

##########Part 1 ###########

'''
    1)  train a KKN classifier for your data  (K = 11). Features: sepal_length,sepal_width,petal_length,petal_width, color
    Hint: pandas.DataFrame.drop() to drop unwanted features
'''

# YOUR CODE GOES HERE

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 11)

modifiedData = modifiedData.drop(columns = ["ID"])

modifiedData2 = modifiedData
modifiedDataOHE = modifiedData
modifiedDataLE = modifiedData
'''
    2)  what is the error in previous part? how you can fix it?
    fix the error and train your model.
    Hint: pandas.get_dummies()
'''
# YOUR CODE GOES HERE

colors = pd.get_dummies(modifiedData['color'])
modifiedData = pd.concat([modifiedData, colors], axis=1)

neigh.fit(modifiedData[["sepal_length","sepal_width","petal_length","petal_width", "blue", "pink", "purple", "red"]], modifiedData["species"])

'''
    3)  test your trained model and evaluate the result for given test set.
'''

X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# YOUR CODE GOES HERE

df = pd.DataFrame(data = X_test)
colors = pd.get_dummies(df[4])
df = pd.concat([df, colors], axis=1)
df = df.drop(columns = [4])

X_test = np.asarray(df)

from sklearn.metrics import classification_report

prediction = (neigh.predict(X_test))
report = classification_report(Y_test, prediction)
#print(report)

'''
    4)  use DictVectorizer from sklearn.feature_extraction to solve Q2
'''
# YOUR CODE GOES HERE

from sklearn.feature_extraction import DictVectorizer


# myDict = (modifiedData2[['color']]).to_dict('records')
myDict = (modifiedData2[['color']]).to_dict('records')
# print(myDict)

vec = DictVectorizer()
x = vec.fit_transform(myDict).toarray()
# print(x)
# print(vec.get_feature_names())
colors = pd.DataFrame.from_dict(x)
feature = vec.get_feature_names()
for i in range(len(feature)):
    feature[i] = feature[i][6:len(feature[i])]
# print(feature)
colors.columns = feature
# print(colors)

modifiedData2 = pd.concat([modifiedData2, colors], axis=1)
neigh.fit(modifiedData[["sepal_length","sepal_width","petal_length","petal_width", "blue", "pink", "purple", "red"]], modifiedData["species"])

'''
    5)  use OneHotEncoder and LabelEncoder from sklearn.preprocessing to solve Q2 and Q3
'''
# # YOUR CODE GOES HERE
# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# data = encoder.fit_transform(modifiedDataOHE[['color']]).toarray()
# modifiedDataOHE = np.concatenate([modifiedDataOHE, data], axis=1)
# # colors = encoder.get_feature_names(['color'])
# colors = encoder.get_feature_names(['color'])
# df_ohe = pd.DataFrame(data = modifiedDataOHE)
# print(colors[0])
# df_ohe.columns = ["sepal_length","sepal_width","petal_length","petal_width", "species", "color", "blue", "pink", "purple", "red"]
# # prints out the label for encoder
# # print(df_ohe)
# neigh.fit(modifiedData[["sepal_length","sepal_width","petal_length","petal_width", "blue", "pink", "purple", "red"]], modifiedData["species"])
# # prints out in terms of numpy, but lose the labels for each column 
# # print(df_ohe)
# prediction = (neigh.predict(X_test))
# report = classification_report(Y_test, prediction)
# # print(report)
# from sklearn.preprocessing import LabelEncoder
# label_enc = LabelEncoder()
# data = label_enc.fit_transform(modifiedDataLE[['color']])
# le = pd.DataFrame({'color_label':data})
# modifiedDataLE = pd.concat([modifiedDataLE, le], axis = 1)
# X_test = np.asarray([[5 , 1, 0.2 , 5,'red'],[0.9 , 7, 6.2 , 2.1,'red'], [0.9 , 7, 6.2 , 2.1,'pink'] , [1.9 , 4, 5 , 0.1,'purple'], [5.9 , 3.3, 0.2 , 2.7,'blue']])
# Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])
# testDF = pd.DataFrame(data = X_test)
# colors = label_enc.fit_transform(testDF[4])
# colorsDF = pd.DataFrame(data = colors)
# testDF = pd.concat([testDF, colorsDF], axis = 1)
# testDF = testDF.drop(columns = [4])
# X_test = np.asarray(testDF)
# neigh.fit(modifiedDataLE[["sepal_length","sepal_width","petal_length","petal_width", "color_label"]], modifiedDataLE["species"])
# prediction = (neigh.predict(X_test))
# report = classification_report(Y_test, prediction)
# # print(report)



##########Part 2 ###########

'''
    1)  Features: sepal_length,sepal_width,petal_length,petal_width.
    use cross validation (CV: Hold-out validation) to find the best value for k.  (try k = 1, 3, 5, 7, 9, 11)
    Hint: use train_test_split from sklearn.model_selection (use 30% of your data as a test set)
    plot the results and pick the best value for k
'''
# YOUR CODE GOES HERE
print("Lab 8")
from sklearn.model_selection import train_test_split
from sklearn import metrics

with open('./data/iris-data.csv') as csvfile:
    dataF = pd.read_csv(csvfile, delimiter=',')

target = dataF["species"]
features = dataF.drop(['species'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.30, random_state=123)
k_list = [1, 3, 5, 7, 9, 11]
accuracy_list = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy_list.append(metrics.accuracy_score(Y_test, Y_pred))
# print(accuracy_list)
big_k = k_list[accuracy_list.index(max(accuracy_list))]
print("Q1, Big K :",big_k)

import matplotlib.pyplot as plt
plt.plot(k_list, accuracy_list, label="Q1 train test split")
plt.title('Cross Validation Methods')

'''
    2)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use KFold from sklearn.model_selection
    plot the results and pick the best value for k
'''
# YOUR CODE GOES HERE
from sklearn.model_selection import KFold

# print(len(dataF))
kf = KFold(n_splits=5, shuffle=True, random_state = 123)

targetAry = target.to_numpy()
featuresAry = features.to_numpy()


for train_index, test_index in kf.split(featuresAry):
    X_train, X_test = featuresAry[train_index], featuresAry[test_index]
    Y_train, Y_test = targetAry[train_index], targetAry[test_index]

k_list = [1, 3, 5, 7, 9, 11]
accuracy_list = []

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    accuracy_list.append(metrics.accuracy_score(Y_test, Y_pred))
# print(accuracy_list)
big_k = k_list[accuracy_list.index(max(accuracy_list))]
print("Q2, Big K :",big_k)
plt.plot(k_list, accuracy_list, 'ro', label = 'Q2 k fold')



'''
    3)  repeat the same question using 5-fold cross validation method to pick the best k.
    Hint: use cross_val_score from sklearn.model_selection
    plot the results and pick the best value for k
'''
# YOUR CODE GOES HERE
from sklearn.model_selection import cross_val_score

k_list = [1, 3, 5, 7, 9, 11]
k_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, features, target, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
    
big_k = k_list[k_scores.index(max(k_scores))]
print("Q3, Big K :",big_k)
# print(max(k_scores))
plt.plot(k_list, k_scores, label = 'Q3 cross val')
plt.legend()

plt.ylabel('accuracy')
plt.xlabel('k\'s')
plt.show()




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

