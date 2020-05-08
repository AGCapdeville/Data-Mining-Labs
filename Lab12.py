
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


ohe = OneHotEncoder(categories = "auto",sparse = False)
# print(len(x_train[0]))
X_train_enc = ohe.fit_transform(x_train)
X_test_enc = ohe.fit_transform(x_test)
# print(X_train_enc.shape)

norm_x_train = tf.keras.utils.normalize(X_train_enc, axis = 1)
norm_x_test = tf.keras.utils.normalize(X_test_enc, axis = 1)

# norm_x_train = tf.keras.utils.normalize(x_train, axis = 1)
# norm_x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))


'''
    2)  Use 'softmax' activation function in output layer, print the predictions/ what is the difference?
'''
# YOUR CODE GOES HERE
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(norm_x_train, y_train, epochs=3)


'''
    3)  save your model as a .h5 (or .hdf5) file
'''
# YOUR CODE GOES HERE
# serialize model to JSON
from keras.models import model_from_json

model_json = model.to_json()
with open("digits_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("digits_model.h5")
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
load_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("BIG 1DICK 3LENGTHS : X ", len(norm_x_test), "whats that dicks shape? :", norm_x_test.shape)
print("BIG 2DICK 4LENGTHS : Y ", len(y_test), "whats that dicks shape? :", y_test.shape )

score = load_model.evaluate(x_test, y_test, verbose=0)
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (load_model.metrics_names[1], score[1]*100))




