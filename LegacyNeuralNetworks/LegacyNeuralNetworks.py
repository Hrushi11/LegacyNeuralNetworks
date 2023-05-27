import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

# 1
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

# 2
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

# 3.
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

# 4
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

#3
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

# 6
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

# 7
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

# 8.
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

# 9.
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
    
# 10.& 11
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

# 12
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

# 13
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