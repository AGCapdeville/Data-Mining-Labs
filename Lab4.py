
'''
Lab 4
'''
print ("Lab 4")

import numpy as np

# Class Mates : Alex & Isaac 

##########Part 0 ###########

'''
    1)  load iris dataset using Pandas

'''

import pandas as pd

with open("./data/iris-data-color.csv") as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')
    
##########Part 1 ###########

'''
    1) using pandas.DataFrame.drop_duplicates drop duplications in "petal_length" feature (keep the last instance) and print out the resulted data
    print out the length of the data set 
    print out the mean of each feature
    print out the frequency count of each label

'''
# YOUR CODE GOES HERE

print("Drop Duplications:")
removedData = data.drop_duplicates("petal_length", keep = 'last')
print(removedData)

print("Length of Removed Duplicates:", len(removedData))

print("Mean of each Feature:\n", removedData.mean(axis = 0),"\n")

print(removedData['species'].value_counts())


    
'''
    2)  plot the data in a single graph
    Hint: pandas.DataFrame.plot()

'''
# YOUR CODE GOES HERE

data.plot()


'''
    3)  plot the bar graph of your data
    Hint: pandas.DataFrame.plot.bar()
    
'''
# YOUR CODE GOES HERE

data.plot.bar()

'''
    4)  plot the histogram graph for "petal_length" feature
    Hint: pandas.DataFrame.plot.histograms()
    
'''
# YOUR CODE GOES HERE
data.plot.hist()

'''
    5)  plot the bar graph to show the frequency of each label 
       
'''
# YOUR CODE GOES HERE

freqGraph = data['species'].value_counts()
freqGraph.plot.bar()

##########Part 2 ###########


'''
    1)  train a KKN classifier for your data  (K = 5)
'''

# YOUR CODE GOES HERE

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier()
neigh.fit(data[["sepal_length","sepal_width","petal_length","petal_width"]], data["species"])


'''
    2)  test your classifier and report the performance 
'''

X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])

# YOUR CODE GOES HERE


'''
    3)  repeat q1 and q2 for "manhattan" distance function
'''

# YOUR CODE GOES HERE




##########Part 3 ###########
'''
    1)  train a KKN classifier for your data  (K = 11)
'''

# YOUR CODE GOES HERE

'''
    2)  test your classifier and report the performance 
'''
# YOUR CODE GOES HERE
X_test = np.asarray([[5 , 1, 0.2 , 5],[0.9 , 7, 6.2 , 2.1], [0.9 , 7, 6.2 , 2.1] , [1.9 , 4, 5 , 0.1], [5.9 , 3.3, 0.2 , 2.7]])
Y_test = np.asarray(['virginica', 'virginica','virginica', 'versicolor' ,'setosa'])


'''
    3)  repeat q1 and q2 for "manhattan" distance function
'''

# YOUR CODE GOES HERE