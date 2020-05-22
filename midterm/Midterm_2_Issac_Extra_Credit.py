'''

(Bonus: 15 points) Use Keras to load one of the state-of-the-art deep CNN models (e.g.:
Inception V4, Inception-ResNet-v2, MobileNetv1, MobileNetv2) trained on the ILSVRC-2012-CLS image classification dataset (ImageNet). Evaluate it’s performance with the given test
set (includes 10 images), and calculate the top1-accuracy and top5-accuracy for that.

'''

import tensorflow as tf
import numpy as np
from keras.applications import inception_resnet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from operator import itemgetter 

rsnet_model = tf.keras.applications.InceptionResNetV2(weights='imagenet')

filename = 'imgs/coffee.jpeg'

original = load_img(filename, target_size=(299, 299))

numpy_image = img_to_array(original)

image_batch = np.expand_dims(numpy_image, axis=0)

processed_image = inception_resnet_v2.preprocess_input(image_batch.copy())

predictions = rsnet_model.predict(processed_image)

accuracy_5 = decode_predictions(predictions)
accuracy_1 = max(accuracy_5, key = itemgetter(1))[0] 

print("Top1-Accuracy:",  accuracy_1)
print("Top5-Accuracy: ",accuracy_5)
