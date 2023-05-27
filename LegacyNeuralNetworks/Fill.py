import os

# String holders for code
activation_function = """
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class ActivationFunction:
    def __init__(self, function_name, grid=False):
        self.function_name = function_name
        self.grid = grid

    def plot(self):
        if self.function_name == 'sigmoid':
            self._plot_sigmoid()
        elif self.function_name == 'relu':
            self._plot_relu()
        elif self.function_name == 'tanh':
            self._plot_tanh()
        elif self.function_name == 'softmax':
            self._plot_softmax()
        elif self.function_name == 'leaky_relu':
            self._plot_leaky_relu()
        else:
            print(f"Unknown activation function: {self.function_name}")

    def _plot_sigmoid(self):
        x = np.linspace(-10, 10, 100)
        y = 1 / (1 + np.exp(-x))
        plt.plot(x, y)
        plt.title('Sigmoid Activation Function')
        plt.xlabel('x')
        plt.ylabel('sigmoid(x)')
        plt.grid(self.grid)
        plt.show()

    def _plot_relu(self):
        x = np.linspace(-10, 10, 100)
        y = np.maximum(0, x)
        plt.plot(x, y)
        plt.title('ReLU Activation Function')
        plt.xlabel('x')
        plt.ylabel('relu(x)')
        plt.grid(self.grid)
        plt.show()

    def _plot_tanh(self):
        x = np.linspace(-10, 10, 100)
        y = np.tanh(x)
        plt.plot(x, y)
        plt.title('Hyperbolic Tangent (tanh) Activation Function')
        plt.xlabel('x')
        plt.ylabel('tanh(x)')
        plt.grid(self.grid)
        plt.show()

    def _plot_softmax(self):
        x = np.linspace(-10, 10, 100)
        e_x = np.exp(x)
        y = e_x / np.sum(e_x)
        plt.plot(x, y)
        plt.title('Softmax Activation Function')
        plt.xlabel('x')
        plt.ylabel('softmax(x)')
        plt.grid(self.grid)
        plt.show()

    def _plot_leaky_relu(self):
        x = np.linspace(-10, 10, 100)
        alpha = 0.2
        y = np.where(x > 0, x, alpha * x)
        plt.plot(x, y)
        plt.title('Leaky ReLU Activation Function')
        plt.xlabel('x')
        plt.ylabel('leaky_relu(x)')
        plt.grid(self.grid)
        plt.show()

# Change activation function names
activation_func = ActivationFunction('tanh')
activation_func.plot()
"""

mcculloh_pitt = """
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class McCullochPittsNeuron:
    def __init__(self):
        pass

    def activate(self, inputs, weights, threshold=3):
        weighted_sum = sum(w * x for w, x in zip(weights, inputs))
        if weighted_sum >= threshold:
            return 1
        else:
            return 0
    
    def andnot(self, x1, x2, weights):
        inputs = np.array([x1, x2])
        return self.activate(inputs, weights) 

neuron = McCullochPittsNeuron()
weights = [1, 4]
x1 = 0; x2 = 0
output = neuron.andnot(x1, x2, weights)
"""

ascii_perceptron = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def activate(self, inputs):
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        if weighted_sum >= 0:
            return 1
        else:
            return 0

    def train(self, X_train, y_train, epochs, learning_rate):
        for _ in range(epochs):
            for inputs, target in zip(X_train, y_train):
                inputs = np.array(inputs)
                prediction = self.activate(inputs)
                error = target - prediction

                # Update weights and bias
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

    def predict(self, X_test):
        predictions = []
        for inputs in X_test:
            inputs = np.array(inputs)
            prediction = self.activate(inputs)
            predictions.append(prediction)
        return predictions

X_train = [[48], [49], [50], [51], [52], [53], [54], [55], [56], [57]]
y_train = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Create Perceptron
input_size = len(X_train[0])
perceptron = Perceptron(input_size)

# Train the Perceptron
epochs = 1000
learning_rate = 0.1
perceptron.train(X_train, y_train, epochs, learning_rate)

# Test the Perceptron
X_test = [[48], [55], [57], [50], [53], [52]]
predictions = perceptron.predict(X_test)

for number, prediction in zip(X_test, predictions):
    result = "Even" if prediction == 0 else "Odd"
    print(f"Number: {number[0]}, Prediction: {result}")
"""

descision_region_perceptron = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class PerceptronPlotter:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.X_train = None
        self.y_train = None
        self.clf = None

    def generate_data(self):
        np.random.seed(self.random_seed)
        self.X_train, self.y_train = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

    def train_classifier(self):
        self.clf = Perceptron().fit(self.X_train, self.y_train)

    def plot_decision_regions(self):
        xx, yy = np.meshgrid(np.arange(self.X_train[:, 0].min() - 1, self.X_train[:, 0].max() + 1, 0.02),
                             np.arange(self.X_train[:, 1].min() - 1, self.X_train[:, 1].max() + 1, 0.02))
        Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=plt.cm.Paired)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Decision Regions')
        plt.show()

    def run(self):
        self.generate_data()
        self.train_classifier()
        self.plot_decision_regions()

plotter = PerceptronPlotter(random_seed=44)
plotter.run()
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

ann_forward_backward = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward_propagation(self, X):
        # Forward propagation through the network
        
        # Layer 1 (input to hidden)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Layer 2 (hidden to output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward_propagation(self, X, y, learning_rate):
        # Backpropagation to update weights and biases
        
        # Calculate gradients
        self.dz2 = self.a2 - y
        self.dW2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        self.dW1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
    
    def train(self, X, y, epochs, learning_rate):
        # Training the neural network
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Backpropagation
            self.backward_propagation(X, y, learning_rate)
            
            # Print loss for every 100 epochs
            if epoch % 100 == 0:
                loss = self.calculate_loss(y, output)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, X):
        # Make predictions using the trained network
        
        output = self.forward_propagation(X)
        predictions = np.argmax(output, axis=1)
        return predictions
    
    def sigmoid(self, x):
        # Sigmoid activation function
        
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function
        
        return x * (1 - x)
    
    def calculate_loss(self, y_true, y_pred):
        # Calculate the mean squared loss
        
        return np.mean(np.square(y_true - y_pred))

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Define the training data (X) and target labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the neural network for 1000 epochs with a learning rate of 0.1
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    output = nn.forward_propagation(X)

    # Backpropagation
    nn.backward_propagation(X, y, learning_rate)

    # Print loss for every 100 epochs
    if epoch % 100 == 0:
        loss = nn.calculate_loss(y, output)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Make predictions on new data
new_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(new_data)

print(predictions)
"""

xor_backprop = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class BackpropagationXOR:
    def __init__(self):
        # Initialize weights and biases
        self.W1 = np.random.randn(2, 2)
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1)
        self.b2 = np.zeros((1, 1))

    def forward_propagation(self, X):
        # Forward propagation through the network

        # Layer 1 (input to hidden)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        # Layer 2 (hidden to output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward_propagation(self, X, y, learning_rate):
        # Backpropagation to update weights and biases

        # Calculate gradients
        self.dz2 = self.a2 - y
        self.dW2 = np.dot(self.a1.T, self.dz2)
        self.db2 = np.sum(self.dz2, axis=0, keepdims=True)
        self.dz1 = np.dot(self.dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        self.dW1 = np.dot(X.T, self.dz1)
        self.db1 = np.sum(self.dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def train(self, X, y, epochs, learning_rate):
        # Training the neural network

        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)

            # Backpropagation
            self.backward_propagation(X, y, learning_rate)

            # Print loss for every 100 epochs
            if epoch % 100 == 0:
                loss = self.calculate_loss(y, output)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        # Make predictions using the trained network

        output = self.forward_propagation(X)
        predictions = np.round(output)
        return predictions

    def sigmoid(self, x):
        # Sigmoid activation function

        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid function

        return x * (1 - x)

    def calculate_loss(self, y_true, y_pred):
        # Calculate the mean squared loss

        return np.mean(np.square(y_true - y_pred))

xor_net = BackpropagationXOR()

# Define the training data (X) and target labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the neural network for 1000 epochs with a learning rate of 0.1
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    output = xor_net.forward_propagation(X)

    # Backpropagation
    xor_net.backward_propagation(X, y, learning_rate)

    # Print loss for every 100 epochs
    if epoch % 100 == 0:
        loss = xor_net.calculate_loss(y, output)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Make predictions on new data
new_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = xor_net.predict(new_data)

# Print the predictions
predictions
"""

art_network = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class ARTNeuralNetwork:
    def __init__(self, num_features, max_categories=100, rho=0.5, beta=1.0):
        self.num_features = num_features
        self.max_categories = max_categories
        self.rho = rho
        self.beta = beta

        self.categories = np.ones((max_categories, num_features))

    def compute_similarity(self, pattern):
        return np.sum(self.categories * pattern, axis=1) / (self.beta + np.sum(self.categories, axis=1))

    def learn(self, patterns):
        for p in patterns:
            pattern = np.array(p)
            while True:
                similarity = self.compute_similarity(pattern)
                if np.max(similarity) < self.rho:
                    # If no category is similar enough, create a new one
                    self.categories = np.vstack((self.categories, pattern))
                    break
                else:
                    # Update the most similar category
                    winner = np.argmax(similarity)
                    self.categories[winner] = self.beta * self.categories[winner] + (1 - self.beta) * pattern
                    if np.sum(self.categories[winner]) / np.sum(pattern) >= self.rho:
                        break
                    else:
                        self.categories = np.delete(self.categories, (winner), axis=0)
        return self.categories

    def predict(self, patterns):
        predictions = []
        for p in patterns:
            pattern = np.array(p)
            similarity = self.compute_similarity(pattern)
            winner = np.argmax(similarity)
            predictions.append(winner)
        return predictions

        
# Define binary patterns
patterns = [
    [1, 0, 0, 1, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
]

# Initialize ART1 network
art1 = ARTNeuralNetwork(num_features=5, max_categories=4, rho=0.6, beta=0.5)

# Learn patterns
categories = art1.learn(patterns)
print(f"Categories after learning: \n{categories}")

# Predict categories for patterns
predictions = art1.predict(patterns)
print(f"Predicted categories: {predictions}")
"""

hopfield_network = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, training_vectors):
        num_vectors = len(training_vectors)
        for vector in training_vectors:
            vector = np.array(vector)
            self.weights += np.outer(vector, vector)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_vectors

    def recall(self, vector, steps=1):
        for _ in range(steps):
            output = np.dot(self.weights, vector)
            vector = np.where(output > 0, 1, -1)
        return vector

# Define binary patterns
training_vectors = [
    [-1, -1, 1, -1, 1, -1, -1, 1],
    [-1, -1, -1, -1, -1, 1, -1, -1],
    [-1, 1, 1, -1, -1, 1, -1, 1],
    [1, 1, -1, 1, -1, 1, 1, -1]
]

# Initialize Hopfield network
hopfield = HopfieldNetwork(num_neurons=8)

# Train network
hopfield.train(training_vectors)

# Try to recall a noisy version of the first training vector
noisy_vector = [-1, -1, 1, -1, 1, 1, -1, 1]  # Flip two bits of the first training vector
recalled_vector = hopfield.recall(noisy_vector, steps=5)

print(f"Noisy input:    {noisy_vector}")
print(f"Recalled output: {recalled_vector.tolist()}")
"""

cnn_object_detection = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class CNNObjectDetection:
    def __init__(self, num_classes=10, filters=32, kernel=(3, 3), dense_nodes=64):
        self.filters = filters
        self.kernel = kernel
        self.dense_nodes = dense_nodes
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters, self.kernel, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.dense_nodes, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        return history

    def plot_accuracy(self, history):
        # Plot accuracy graph
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_loss(self, history):
        # Plot loss graph
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, batch_size=32, plot=False):
        history = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        if plot: self.plot_accuracy(history)
        if plot: self.plot_loss(history)

        self.evaluate_model(X_test, y_test)

        
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Create and train the model
cnn = ConvNetImageClassification(num_classes=10)
cnn.run(X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        epochs=20, batch_size=128)
"""

cnn_image_classification = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class CNNObjectDetection:
    def __init__(self, num_classes=10, filters=32, kernel=(3, 3), dense_nodes=64):
        self.filters = filters
        self.kernel = kernel
        self.dense_nodes = dense_nodes
        self.num_classes = num_classes
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(self.filters, self.kernel, activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.dense_nodes, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        return history

    def plot_accuracy(self, history):
        # Plot accuracy graph
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_loss(self, history):
        # Plot loss graph
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=10, batch_size=32, plot=False):
        history = self.train_model(X_train, y_train, X_val, y_val, epochs, batch_size)

        if plot: self.plot_accuracy(history)
        if plot: self.plot_loss(history)

        self.evaluate_model(X_test, y_test)

# Load and preprocess the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                  test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Create and train the model
cnn = ConvNetImageClassification(num_classes=10, filters=32, kernel=(3, 3), dense_nodes=64)
cnn.run(X_train, y_train, 
        X_val, y_val, 
        X_test, y_test, 
        epochs=20, batch_size=128,
        plot=True)
"""

cnn_tf_implementation = """
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class CNNModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=128):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def predict(self, X):
        return self.model.predict(X)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

num_classes = 10
cnn_model = CNNModel(num_classes)
cnn_model.train(X_train, y_train, epochs=10, batch_size=32)
cnn_model.evaluate(X_test, y_test)
predictions = cnn_model.predict(X_test)
"""

mnist_detection = """ 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

class MNISTClassifier:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        return model

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    def visualize_data(self, X_data, y_data, num_samples=5):
        fig, axes = plt.subplots(1, num_samples, figsize=(10, 4))

        for i in range(num_samples):
            axes[i].imshow(X_data[i], cmap='gray')
            axes[i].set_title(f"Label: {y_data[i]}")
            axes[i].axis('off')

        plt.show()

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create an instance of the MNISTClassifier
mnist_classifier = MNISTClassifier()

# Train the model
mnist_classifier.train(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
mnist_classifier.evaluate(X_test, y_test)

# To visualize the data
mnist_classifier.visualize_data( X_test, y_test, num_samples=5)
"""


masterDict = {
    'activation_function' : activation_function,
    'mcculloh_pitt': mcculloh_pitt,
    'ascii_perceptron': ascii_perceptron,
    'descision_region_perceptron': descision_region_perceptron,
    'recognize_5x3_matrix': recognize_5x3_matrix,
    'ann_forward_backward': ann_forward_backward,
    'xor_backprop': xor_backprop,
    'art_network': art_network,
    'hopfield_network':hopfield_network,
    'cnn_object_detection': cnn_object_detection,
    'cnn_image_classification': cnn_image_classification,
    'cnn_tf_implementation': cnn_tf_implementation,
    'mnist_detection': mnist_detection   
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