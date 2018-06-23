# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

#CONVOLUTION---------------------------------
# 32 - number of filters
# (3,3) - dimension of a filter image 3x3
# inputshape = image of 64x64, 3 = RGB 
# activation = relu => ???????
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

#POOLING------------------------------------
#reducing the number of nodes for the upcoming layers
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#FLATTENING---------------------------------
#taking the 2D array of pooled pixels and combine them to a one dimensional single vector
classifier.add(Flatten())

#Add a fully connected layer
classifier.add(Dense(units = 16, activation = 'relu'))

#OUTPUT LAYER
#only one node, because it is the root of the binary tree of decisions
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#preparing the image dataset for proccessing 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_set = test_datagen.flow_from_directory('testset', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

#fit the data to our model!
#steps per epoch contains the number of training images
#epoch - a training cycle
classifier.fit_generator(training_set, steps_per_epoch = 69, epochs = 25, validation_data = test_set, validation_steps = 10)

# serialize model to JSON
classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")