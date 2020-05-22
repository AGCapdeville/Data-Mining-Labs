'''
(Bonus: 15 points) Use Keras to load one of the state-of-the-art deep CNN models (e.g.:
Inception V4, Inception-ResNet-v2, MobileNetv1, MobileNetv2) trained on the ILSVRC-2012-CLS 
image classification dataset (ImageNet). Evaluate it’s performance with the given test
set (includes 10 images), and calculate the top1-accuracy and top5-accuracy for that.
'''

import tensorflow as tf
import numpy as np

from keras.applications import mobilenet_v2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from operator import itemgetter 

selected_img = 'imgs/lion.jpeg'

mobilenet_model = tf.keras.applications.MobileNetV2(weights='imagenet')

original = load_img(selected_img, target_size=(224, 224))
numpy_image = img_to_array(original)
image_batch = np.expand_dims(numpy_image, axis=0)

procc_input = mobilenet_v2.preprocess_input(image_batch.copy())
mobilenet_model = mobilenet_model.predict(procc_input)
decode = decode_predictions(mobilenet_model)
acc = max(decode, key = itemgetter(1))[0] 

print("\n\nFor picture: ", selected_img)
print(50*'=')
print("Top 1 Accuracy:\n", (acc))
print(50*'=')
print("Top 5 Accuracy:\n",decode)
print(50*'=')

