
'''
Lab 13
'''
print ("Lab 13")

##########Part 0 ###########
'''
    1)  from sklearn.datasets import load_digits  (Each datapoint is a 8x8 image of a digit)
    Split your data into train(80% of data) and test(20% of data) via random selection
'''
# YOUR CODE GOES HERE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
(data, target) = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)

##########Part 1 ###########
'''
    1)  Try RandomForestClassifier sklearn.ensemble. Try different number of trees and criterion to get a reasonable classifier. (use holdout validation)
'''
# YOUR CODE GOES HERE
from sklearn.ensemble import RandomForestClassifier
# TODO: Use hold out.
randForestC = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=42)

'''
    2)  print classification report for the test set
'''
# YOUR CODE GOES HERE
from sklearn.metrics import classification_report

randForestC.fit(x_train, y_train)
y_pred = randForestC.predict(x_test)
print(classification_report(y_test, y_pred))


##########Part 2 ###########

'''
    1)  Try AdaBoostClassifier from sklearn.ensemble
    
    Hint: First you have to define a week classifier (e.g.: KNN, linear LR, Linear SVM)
    Try different base classifier to get a reasonable model. (use holdout validation)
       
'''

# YOUR CODE GOES HERE




'''
    2)  print classification report for the test set
'''
# YOUR CODE GOES HERE



##########Part 3 ###########

'''
    1) Lets get back to Keras again!
    load mnist dataset from Keras:  from keras.datasets import mnist
     Try to  build a CNN in Keras. You can start from a very simple model with only one 
    Conv2D layer, one Maxpooling 2D, Flatten layer and one FC(Dense) layer.


    
    activation for Conv2D: ReLU, kernel size: 3, 30 kernels
    activation fcn for output layer: softmax
    metrics: 'accuracy'
    loss: 'categorical_crossentropy'
    validation_split = 0.3
    
        
'''

# YOUR CODE GOES HERE


'''
    2)  print classification report for the test set
'''
# YOUR CODE GOES HERE


'''
    3)  Research about Dropout layer and try to add a dropout layer with ratio of 0.5 and see the result.
'''
# YOUR CODE GOES HERE