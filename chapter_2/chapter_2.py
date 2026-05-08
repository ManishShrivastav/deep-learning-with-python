# import MNIST dataset from keras

from keras.datasets import mnist

# load the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print the shape of the training images (60000, 28, 28) -> 60000 images, each of size 28x28 pixels
print(train_images.shape)

# print the shape of the training labels (60000,) -> 60000 labels, each corresponding to an image
print(train_labels.shape)

# print the shape of the test images (10000, 28, 28) -> 10000 images, each of size 28x28 pixels
print(test_images.shape)

# print the shape of the test labels (10000,) -> 10000 labels, each corresponding to an image
print(test_labels.shape)

import keras
from keras import layers

# adding 2 layers to the model, these are dense layers (also known as fully connected layers), 
# the first layer has 512 neurons and uses the ReLU activation function, the second layer has 
# 10 neurons and uses the softmax activation function. The softmax activation function is used in the output 
# layer of a classification model to convert the output into a probability distribution over the 10 classes (digits 0-9).

model = keras.Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compile the model with the Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric. 
# The Adam optimizer is an efficient optimization algorithm that adjusts the learning rate during training. 
# The sparse categorical cross entropy loss function is used for multi-class classification problems where the 
# labels are integers (not one-hot encoded).
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# reshape the training and test images from (60000, 28, 28) and (10000, 28, 28) to (60000, 784) and (10000, 784) respectively.
# This is done to convert the 2D images into 1D vectors that can be fed into the dense layers of the model. 
# The pixel values are also normalized to the range [0, 1] by dividing by 255, which helps the model to 
# converge faster during training.
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# train the model for 5 epochs with a batch size of 128. An epoch is one complete pass through 
# the entire training dataset, and a batch is a subset of the training data that is used to update 
# the model's weights during each iteration of training.  
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# predict the labels for the first 10 test images using the trained model. 
# The predict method returns an array of shape (10, 10), where each row corresponds 
# to a test image and each column corresponds to a class (digit 0-9). 
# The values in the array represent the predicted probabilities for each class, and the class 
# with the highest probability is the predicted label for that image.
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]

# returns the index of the highest predicted probability, which corresponds to the predicted label for the first test image.
predictions[0].argmax() 
predictions[0][7] # predicted probability for the class '7' for the first test image

test_labels[0] # actual label for the first test image, which is '7'

# evaluate the model on the test dataset to get the test loss and test accuracy. 
# The evaluate method returns the loss value and the metric value (accuracy in this case) 
# for the test dataset. The test accuracy is printed to the console.
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

