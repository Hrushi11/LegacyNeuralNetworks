import os

# String holders for code
activation_function_1 = """
#ann 1 -->  Write a Python program to plot a few activation functions that are being used in neural networks. 
import numpy as np 
import matplotlib.pyplot as plt
def sigmoid(x):
    return  1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def tanh(x):
    return np.tanh(x)
x= np.linspace(-10,10,100)
plt.Figure(figsize=(10,6))
plt.plot(x,sigmoid(x),label ="sigmoid")
plt.plot(x,relu(x),label ="relu")
plt.plot(x,tanh(x),label ="tanh")
plt.show()
"""
mcculloh_pitt_2 ="""
# ann 02 -->  Generate ANDNOT function using McCulloch-Pitts neural net by a python program
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 3, 1]
w2 = [1, 1, 2, 1]
t = 3
#output
print("x1    x2    w1    w2    t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)
"""
ascii_perceptron_3 = """ 
#ann 03 --> Write a Python Program using Perceptron Neural Network to recognise even and odd numbers. Given numbers are in ASCII form 0 to 9
from sklearn.linear_model import Perceptron
import numpy as np
X = np.array([ [48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
 [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
])
w = np.zeros(X.shape[0])
b = 0
def activation(z):
 return 1 if z >= 0 else 0
for epoch in range(1000):
 for x, y in zip(X[0], X[1]):
 z = np.dot(w, X[:, x-48]) + b
 y_hat = activation(z)
 error = y - y_hat
 w += error * X[:, x-48]
 b += error
while True:
 x = input("Enter a number between 0 and 9: ")
 if not x.isdigit() or int(x) < 0 or int(x) > 9:
 print("Invalid input!")
 z = np.dot(w, X[:, int(x)]) + b
 y_hat = activation(z)
 print("(Number is Even" if y_hat == 0 else "Number is Odd")
"""
descision_region_perceptron_4 = """ 
#ann 6/4-->Implement perceptron learning law with its decision regions using python. Give the output in graphical form 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
# Generate some random data
np.random.seed(0)
X = np.random.randn(100, 2)
y = np.where(X[:,0] + X[:,1] > 0, 1, -1)
# Train the perceptron classifier
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
# Plot the decision boundary
xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1),np.arange(ymin, ymax, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()
"""

bam = """ 
#ann 4-->In this program, the bam function implements the Bidirectional Associative Memory (BAM) algorithm
import numpy as np
def bam(input_patterns,output_patterns):
    input_patterns=np.array(input_patterns)
    output_patterns= np.array(output_patterns)
    weight_matrix = np.dot(output_patterns.T,input_patterns)
    def activation(input_pattern):
        output_pattern = np.dot(weight_matrix,input_pattern)
        output_pattern[output_pattern>=0]=1
        output_pattern[output_pattern<0]=-1
        return output_pattern    
    print("input patterns | output patterns")
    for i in range(input_patterns.shape[0]):
        input_pattern = input_patterns[i]
        output_pattern = activation(input_pattern)
        print(f"{input_pattern} | {output_pattern}")
input_patterns=[[1,-1,1,-1],[1,1,-1,-1]]
output_patterns=[[1,1],[-1,-1]]
bam(input_patterns,output_patterns)
"""

ann_forward_backward = """ 
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def train_neural_network(X, y, hidden_size, epochs, learning_rate):
    input_size, output_size = X.shape[1], y.shape[1]
    weights1 = np.random.randn(input_size, hidden_size)
    weights2 = np.random.randn(hidden_size, output_size)
    for _ in range(epochs):
        # Forward propagation
        hidden_layer = sigmoid(X.dot(weights1))
        output_layer = sigmoid(hidden_layer.dot(weights2))
        # Back propagation
        output_error = y - output_layer
        output_delta = output_error * sigmoid_derivative(output_layer)
        hidden_error = output_delta.dot(weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer)
        # Update the weights using gradient descent
        weights2 += learning_rate * hidden_layer.T.dot(output_delta)
        weights1 += learning_rate * X.T.dot(hidden_delta)
    # Print the final loss
    loss = np.mean(np.square(y - output_layer))
    print("Final Loss:", loss)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
train_neural_network(X, y, hidden_size=2, epochs=10000, learning_rate=0.1)

"""

xor_backprop_7 = """ 
#ann 07 ->Implement to show Back Propagation Network for XOR function with Binary Input and Output
import numpy as np
import math
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(42)
synapse_0 = 2 * np.random.random((2, 3)) - 1
synapse_1 = 2 * np.random.random((3, 1)) - 1
for i in range(10000):
    # Forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    # Calculate the error
    layer_2_error = y - layer_2
    # Back propagation
    layer_2_delta = layer_2_error * sigmoid_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    layer_1_delta = layer_1_error * sigmoid_derivative(layer_1)
    # Update the weights
    synapse_1 += layer_1.T.dot(layer_2_delta)
    synapse_0 += layer_0.T.dot(layer_1_delta)  #T ==> transpose

print("Output after training:")
print(layer_2)

"""

art_network_8 = """ 
# ann 08 --> Write a python program to illustrate ART neural network

import tensorflow as tf
import numpy as np
X = np.array([
    [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0],
    [1, 0, 0], [0, 0, 0], [1, 1, 1],
])
y = np.array([0, 1, 2, 0, 2, 1, 0, 2])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dense(3, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=200)
X_test = np.array([
    [1, 1, 1], [0, 0, 0], [1, 0, 1],
])
y_pred = np.argmax(model.predict(X_test), axis=1)
print(y_pred)

"""

hopfield_network_9 = """ 
# ann 09 --> Write a python program to design a Hopfield Network which stores 4 vectors. 
import numpy as np
# Define the patterns to be stored
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])
# Train the Hopfield network
weights = np.dot(patterns.T, patterns)
np.fill_diagonal(weights, 0)
# Recall a pattern
recall_pattern = np.array([1, 1, -1, -1])
retrieved_pattern = np.sign(np.dot(weights, recall_pattern))
print("Retrieved Pattern:")
print(retrieved_pattern)

"""

cnn_object_detection_10 = """ 
#ann 10 -->Write python program to implement CNN object detection . Discuss numerous performance evalution metrics for evaluating the object detecting algorithm performace
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights='imagenet')
# Load image and preprocess it
image = tf.keras.preprocessing.image.load_img('download.jpeg', target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
image = tf.expand_dims(image, axis=0)
# Run object detection
predictions = model.predict(image)
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
# Print top predicted objects
for _, label, confidence in decoded_predictions[0]:
    print(f"{label}: {confidence * 100}%")
"""

cnn_image_classification_11 = """ 
#ann 11 -->Implement an image classification challenge, create and train a ConvNet in Python using TensorFlow
import tensorflow as tf
# Load and preprocess the dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
# Define the ConvNet architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

"""

cnn_tf_implementation_12 = """
# ann 12 -->   Implement TensorFlow/Pytorch implementation of CNN
import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# Flatten the output of the previous layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
"""

mnist_detection_13 = """ 
# ann 13 --> handw 
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Load MNIST dataset from scikit-learn
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
# Convert target values to integers
y = y.astype(int)
# Preprocess the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
model.save('project.h5')
"""


recognize_5x3_matrix = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
class PerceptronNN:
    def __init__(self, nn=10):
        self.nn = nn
        self.clf = MLPClassifier(hidden_layer_sizes=(self.nn,), random_state=42)
        self.train_data = {
            0: [[1, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]],
            1: [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            2: [[1, 1, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 1]],
            3: [[1, 1, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
            4: [[1, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]],
            5: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]],
            6: [[1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            7: [[1, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            8: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 1], [1, 1, 1]],
            9: [[1, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]]
        }
    def train(self):
        # Create the training set
        training_data = self.train_data
        X_train = []
        y_train = []
        for digit, data in training_data.items():
            X_train.append(np.array(data).flatten())
            y_train.append(digit)

        # Convert training data to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # print(X_train, y_train)
        # Train the MLP classifier
        self.clf.fit(X_train, y_train)
    def recognize(self, test_data):
        # Convert test data to NumPy array
        X_test = np.array(test_data)
        predictions = self.clf.predict(X_test)
        majority_vote = np.argmax(np.bincount(predictions))
        return majority_vote
recognizer = PerceptronNN(16)
recognizer.train()
test_data = np.array([test_data]).flatten()
print(predictions)
"""

masterDict = {
    'activation_function' : activation_function_1,
    'mcculloh_pitt': mcculloh_pitt_2,
    'ascii_perceptron': ascii_perceptron_3,
    'descision_region_perceptron': descision_region_perceptron_4,
    'bam': bam,
    'ann_forward_backward': ann_forward_backward,
    'xor_backprop': xor_backprop_7,
    'art_network': art_network_8,
    'hopfield_network':hopfield_network_9,
    'cnn_object_detection': cnn_object_detection_10,
    'cnn_image_classification': cnn_image_classification_11,
    'cnn_tf_implementation': cnn_tf_implementation_12,
    'mnist_detection': mnist_detection_13,
    'recognize_5x3_matrix':recognize_5x3_matrix  
}

class Writer:
    def __init__(self, filename):
        self.filename = os.path.join(os.getcwd(), filename)
        self.masterDict = masterDict
        self.questions = list(masterDict.keys())

    def getCode(self, input_string):
        input_string = self.masterDict[input_string]
        with open(self.filename, 'w') as file:
            file.write(input_string)
        print(f'##############################################')

if __name__ == '__main__':
    write = Writer('output.txt')
    # print(write.questions)
    write.getCode('descision_region_perceptron')
