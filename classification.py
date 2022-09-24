# https://www.tensorflow.org/tutorials/keras/classification
# Imports
import tensorflow as tf 
from tensorflow import keras

#helper libraries
import numpy as np 
import matplotlib.pyplot as plt

#Datasets - 60,000 traninng and 10,000 valiation/testing images
fashion_minst =  keras.datasets.fashion_mnist #load dataset
#split into train and test
(train_images, train_labels), (test_images, test_labels) = fashion_minst.load_data()

train_images.shape # (60000, 28, 28) 60,000 images made up of 28X28 pixels (784 in total)
type(train_images) #numpy.ndarray

# look at one pixel
train_images[0, 23, 23]  #194
# pixel values between 0 ad 255 , 0 is black and 255 is white, means greyscale image no color

# look first 10 training labels
train_labels[:10]  # array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)

#labels from 0-0, each integer represents a specific article of clothing
# create an array of label names to indicate which is which
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#  look at some of these images
plt.figure()
plt.imshow(train_images[6])
plt.colorbar()
plt.grid(False)
plt.show()

#Data Processing - last step before creating model is to pre process out data
# applying some prior transformations to data before feeding it the model
# scale all out greyscale pixel values (0-255) to be between 0 and 1
# do this by divding each value in the training and testing sets by 255.0
# because smaller values will make ti easier for the model to process our values

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Build the model
# use a keras sequestial model withe 3 different layers. 
# this model represents a feed-forward neural network (one that passes values from left to right)
# break down each layer and its architecure below

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #layer 1
    tf.keras.layers.Dense(128, activation='relu'), #layer 2
    tf.keras.layers.Dense(10)                      #layer 3
])

#Compile the model
# Loss function — it measures how accurate the model is during training. 
# You want to minimize this function to "steer" the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. 

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model - fitting it to the training data
model.fit(train_images, train_labels, epochs=10) # accuracy of 91%. 
# this is the accuracy on testing or training data. So now if we want to find what
#Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# 88.5% accurate compare to test accuracy (because already memorize, which means overfit  model.
# changing some of this architecture, the optimizer the loss function, epochs eight – to improve better accuracy

test_images.shape #(10000, 28, 28)

# Make predictions
# predicting array of images 
predictions = model.predict(test_images) # single image predict [test_images[0]]
print(predictions) # predictions[0]
#see the prediction for the probability distribution that was calculated on output layer for

#the model has predicted the label for each image in the testing set. look at the first prediction:
predictions[0]
#array([1.3835326e-08, 2.7011181e-11, 2.6019606e-10, 5.6872784e-11,
#       1.2070331e-08, 4.1874609e-04, 1.1151612e-08, 5.7000564e-03,
#       8.1178889e-08, 9.9388099e-01], dtype=float32)

# figure out what class this is predicting for using  argmax
# A prediction is an array of 10 numbers. 
np.argmax(predictions[0]) # 9

# print the class name and image (angel boot)
print(class_names[np.argmax(predictions[0])]) 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Verify predictions

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# look at the 0th image, predictions, and prediction array. 
# Correct prediction labels are blue and incorrect prediction labels are red. 
#The number gives the percentage (out of 100) for the predicted label.

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
