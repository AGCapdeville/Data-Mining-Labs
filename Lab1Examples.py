'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization
from keras.optimizers import RMSprop,Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.models import load_model



batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

####################################################################################################
# This example shows batch normalization rather than dropout layers and using different optimizers...
#early_stopping = EarlyStopping(monitor='val_loss', patience=2)
loss1 = 'categorical_crossentropy'
loss2 = 'mean_squared_error'

epochs =20

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss=loss2,
              optimizer=Adadelta(),
              metrics=['accuracy'])
tbCallBack = keras.callbacks.ModelCheckpoint(filepath='/home/mharmon/',period=1)
tbCallBack.set_model(model)
for i in range(epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=0,
                        validation_data=(x_test, y_test))
    print(history.history)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################################################################################

####################################################################################################
# This example shows earlystopping (which is probably the most important callback. We also save only the best model here.
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
check = keras.callbacks.ModelCheckpoint('/home/mharmon/DeepLearning/model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
hist = keras.callbacks.History()
loss1 = 'categorical_crossentropy'


epochs =200

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss=loss1,
              optimizer=Adadelta(),
              metrics=['accuracy'])
tbCallBack = keras.callbacks.ModelCheckpoint(filepath='/home/mharmon/',period=1)
tbCallBack.set_model(model)
history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),callbacks=[early_stopping,check])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
del model
##############################################################################################################

####################################################################################################
# If we want to load this model back up, we do the following:
model = load_model('model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##############################################################################################################
# A few other things you can do with the model

predictions = model.predict(x_test, batch_size=32, verbose=0)
eval = model.evaluate(x_test, y_test, batch_size=32, verbose=0, sample_weight=None)
classes = model.predict_classes(x_test, batch_size=32, verbose=0)
probs = model.predict_proba(x_test, batch_size=32, verbose=0)

#############################################################################################################
