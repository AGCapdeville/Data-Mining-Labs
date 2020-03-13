
'''
Lab 3
'''
print ("Lab 3")

##########Part 1 ###########

'''
    1) Write a python function to find the frequency of the charchters in a sentence
    Hint : Use a dictionary
    e.g.: "hello"
    {"h":1, "e":1, "l":2, "o":1}

'''
# YOUR CODE GOES HERE

import pandas as pd




######### Part 2 ###########


'''
    1) using pandas.read_csv load and print out the iris-data.csv

'''
# YOUR CODE GOES HERE

with open("./data/iris-data-color.csv"
) as csvfile:
    data = pd.read_csv(csvfile, delimiter=',')
print(data)

'''
    2) after loading the data, print out index, columns, and values information and their types (in a comment line expalin what are they)
       print out the length of the data set 
       print out the last 50 data points
       print out the labels 
       print out the labels and petal_length

slide notes:        
       print(type(data))
       print(data.index)
       print(data.columns)
       print(data.values)
       print(data.values[100])
       print(data.values[1:6])

'''
# YOUR CODE GOES HERE
print("\nINDEX: \n", data.index)
print("\nCOLUMNS: \n", data.columns)
print("\nDATA: \n", data.values)
print("\nLENGTH: ", len(data))
print()
print("last 50 data points:", data[-50:])
print("\nLABELS:")
print(data["species"])
print("\nLABELS + petal_length:")
print(data[["species","petal_length"]])
'''
    3) print out the mean and std of each feature
    print out the mean of petal_length for first 100 samples
    print out the maximum and minimum values for each feature

'''
# YOUR CODE GOES HERE

import statistics as stat

print("\nsepal length  mean:",stat.mean(data["sepal_length"]))
print("             stdev:",stat.stdev(data["sepal_length"]))
        
print("\nsepal width   mean:",stat.mean(data["sepal_width"]))
print("             stdev:",stat.stdev(data["sepal_width"]))
        
print("\npetal length  mean:",stat.mean(data["petal_length"]))
print("             stdev:",stat.stdev(data["petal_length"]))
        
print("\npetal width   mean:",stat.mean(data["petal_width"]))
print("             stdev:",stat.stdev(data["petal_width"]))

print("\nmean of first 100 samples of petal length:",stat.mean(data["petal_length"].head(100)))

print("\nsepal length  min:",min(data["sepal_length"]))
print("              max:",max(data["sepal_length"]))
        
print("\nsepal width   min:",min(data["sepal_width"]))
print("              max:",max(data["sepal_width"]))
        
print("\npetal length  min:",min(data["petal_length"]))
print("              max:",max(data["petal_length"]))
        
print("\npetal width   min:",min(data["petal_width"]))
print("              max:",max(data["petal_width"]))


'''
    4)  print out the frequency count of each label
    Hint: use pandas’ function value_counts 

'''
# YOUR CODE GOES HERE

print("\nFreq. count of Sepal Lengths:\n",pd.value_counts(data["sepal_length"]))
print("\nFreq. count of Sepal Widths:\n",pd.value_counts(data["sepal_width"]))
print("\nFreq. count of Petal Lengths:\n",pd.value_counts(data["petal_length"]))
print("\nFreq. count of Petal Widths:\n",pd.value_counts(data["petal_width"]))

######### Part 3 ###########

'''
    1) using pandas.DataFrame.drop_duplicates drop duplications in "petal_length" 
    feature (keep the last instance) and print out the resulted data
    
    print out the length of the data set 
    print out the mean of each feature
    print out the frequency count of each label

'''
# YOUR CODE GOES HERE
dup = data["petal_length"].drop_duplicates(keep='last')
print("\nPetal lengths w/out duplicates keeping the last:")
print(dup)
print("\n length:",len(dup))
print("\n mean:",stat.mean(dup))
print("\n freq:")
print(pd.value_counts(dup))
'''
    2)  plot the data in a single graph
    Hint: pandas.DataFrame.plot

'''
# YOUR CODE GOES HERE
print("\nGraph of the data:")
data.plot()





