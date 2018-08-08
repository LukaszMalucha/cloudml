### Convolutional Neural Network  - Cat or Dog ???





##################################################################### Building the CNN ################################################



### Importing Libraries

from keras.models import Sequential     # initialize
from keras.layers import Conv2D         # saving model
from keras.models import load_model   # Convolution layer
from keras.layers import MaxPooling2D   # Pooling
from keras.layers import Flatten        # Flattening
from keras.layers import Dense          # Fully connected layers for ANN


# from IPython.display import display
# from PIL import Image
import h5py
import os
from tensorflow.python.lib.io import file_io

### Initializing the CNN

classifier = Sequential()

### Step 1 - Convolution 32x3x3

classifier.add(Conv2D(32, (3, 3),                            ## 32 - number of feature detectors, 3-rows and 3-cols                  
                             input_shape=(96, 96, 3),      ## fixed size of image to standardize dataset + 3d array for color img (reverse order for Tensorflow backend)
                             activation = 'relu'))         ## to make sure there is no negative values in pixel maps to have non-linearity


### Step 2 - Max Pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))            ## dims 2x2

### Step 2b - add additional convolutional layer for better result (from 50% to 80% accuracy)

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))      ## No input shape as it was already done
classifier.add(MaxPooling2D(pool_size = (2,2)))            ## dims 2x2

### Step 3 - Flattening to one single vector

classifier.add(Flatten())

### Step 4 - Full connection

# Hidden layer - 128 as a experience guess

classifier.add(Dense(activation = 'relu', units = 128))

# Output layer - one node

classifier.add(Dense(activation = 'sigmoid', units = 1))  ## softmax for non--binary


### Compile the CNN Model

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



############################################################ Fitting the CNN to the images ###########################################

### From https://keras.io/preprocessing/image/   apply some random transformations on image dataset
### flow_from_directory method code from webpage

from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.platform import gfile

# def read_path(gcs_path):
#   path = file_io.FileIO(gcs_path)
#   return path
   
training_data = gfile.ListDirectory('gs://image-classifier-mlengine/dataset/training_set/')
test_data =  gfile.ListDirectory('gs://image-classifier-mlengine/dataset/test_set/')

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(training_data,       ## extraction directory
                                                 target_size=(96, 96),         ## same dims as in cnn
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(test_data,                ## extraction directory
                                            target_size=(96, 96),              ## same dims as in cnn
                                            batch_size=32,
                                            class_mode='binary')

### FITTING CLASSIFIER

classifier.fit_generator(training_set,
                         steps_per_epoch=10,            ## train images
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=10)            ## test images




classifier.save('my_classifier.h5')        


with file_io.FileIO('my_classifier.h5', mode='rb') as input_f:
    with file_io.FileIO('my_classifier.h5', mode='wb+') as output_f:
        output_f.write(input_f.read())








