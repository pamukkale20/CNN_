# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import TensorBoard
from sklearn import metrics
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import keras 
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('drive/LeafImg_V2/train',
                                                 target_size = (100, 100),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('drive/LeafImg_V2/test',
                                            target_size = (100, 100),
                                            batch_size = 1,
                                            class_mode = 'binary')

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a forth convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a fifth convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dense(128))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1))
classifier.add(Activation('sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
classifier.fit_generator(training_set,
                         samples_per_epoch = 719,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 179)
classifier.save("CNN_1.model")

import cv2
import tensorflow as tf

CATEGORIES = ["A_healty","B_illness"]

def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
#model = tf.keras.models.load_model("64x3-CNNleaf2.model")
filepath2="CNN_1.model"
model=tf.keras.models.load_model(
    filepath2,
    custom_objects=None,
    compile=True
)
#prediction = model.predict([prepare('PetImages/healthy/IMG_20190211_191727 re.jpg')])
pathimg='DSC_1534.jpg'
prediction = model.predict([prepare(pathimg)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
img = cv2.imread(pathimg,1)
img=cv2.resize(img, (300, 300))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
