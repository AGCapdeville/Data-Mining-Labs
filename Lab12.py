'''
Lab 12
'''
print ("Lab 12")

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
    1)  Try MLPClassifier from sklearn.neural_network
        (a NN with two hidden layers, each with 100 nodes)
'''
# YOUR CODE GOES HERE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

mlp_clf = MLPClassifier(hidden_layer_sizes=(100,100,), max_iter=300,activation = 'relu',solver='adam',random_state=1)
mlp_clf.fit(x_train, y_train)
y_pred = mlp_clf.predict(x_test)

'''
    2)  print classification report for the test set
'''
# YOUR CODE GOES HERE
print(classification_report(y_test, y_pred))

##########Part 2 ###########
'''
    1)  Try to have the same NN in Keras. Try different activation functions for hidden layers to get a reasonable network.
        Hint: you need to convert your labels to vectors of 0s and 1  (Try OneHotEncoder from sklearn.preprocessing)

    activation fcn for output layer: sigmoid
    metrics: 'accuracy'
    loss: 'categorical_crossentropy'
    validation_split = 0.3
'''
# epochs from 5-10
# predict, x, batchsize
# YOUR CODE GOES HERE

import numpy as np
import tensorflow as tf
# from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import load_model
import h5py

norm_x_train = tf.keras.utils.normalize(x_train, axis = 1)
norm_x_test = tf.keras.utils.normalize(x_test, axis = 1)

print(50*"=")
print("PART 2 : Q1 NN with activation: relu")

model_0 = tf.keras.models.Sequential()
model_0.add(tf.keras.layers.Flatten())
model_0.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model_0.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_0.fit(norm_x_train, y_train, batch_size=32, epochs=10)

val_loss, val_acc = model_0.evaluate(norm_x_test, y_test)
print(40*"=")
print("activation = relu:")
print("loss:",val_loss, ", acc:",val_acc)
print(40*"=")

print(50*"=")
print("PART 2 : Q1 NN with activation: sigmoid")
model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Flatten())
model_1.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_1.fit(norm_x_train, y_train, batch_size=32, epochs=10)

val_loss, val_acc = model_1.evaluate(norm_x_test, y_test)
print(40*"=")
print("activation = sigmoid:")
print("loss:",val_loss, ", acc:",val_acc)
print(40*"=")


'''
    2)  Use 'softmax' activation function in output layer, print the predictions/ what is the difference?
'''
# YOUR CODE GOES HERE
print(50*"=")
print("PART 2 : Q2 NN with Softmax")
model_2 = tf.keras.models.Sequential()
model_2.add(tf.keras.layers.Flatten())
model_2.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_2.fit(norm_x_train, y_train,batch_size=32, epochs=10)

val_loss, val_acc = model_2.evaluate(norm_x_test, y_test)
print(40*"=")
print("activation = softmax:")
print("loss:",val_loss, ", acc:",val_acc)
print(40*"=")

print(50*"=")


'''
    3)  save your model as a .h5 (or .hdf5) file
'''
# YOUR CODE GOES HERE
# serialize model to JSON
from keras.models import model_from_json

model_json = model_2.to_json()
with open("digits_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_2.save_weights("digits_model.h5")
print("Saved model to disk")


'''
    4)  load your saved your model and test it using the test set 
'''
# returns a compiled model
# identical to the previous one
 
# load json and create model
json_file = open('digits_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
load_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
load_model.load_weights("digits_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
load_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

val_loss, val_acc = load_model.evaluate(norm_x_test, y_test)
print(40*"=")
print("Loaded Model: ( model_2 )")
print("loss:",val_loss, ", acc:",val_acc)
print(40*"=")




