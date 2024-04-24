import numpy as np
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
def setParameters(X, Y, hidden_size):
    np.random.seed(3)
    input_size = X.shape[0] # number of neurons in input layer
    output_size = Y.shape[0] # number of neurons in output layer.
    W1 = np.random.randn(hidden_size, input_size)*np.sqrt(1/input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size)*np.sqrt(1/hidden_size)
    b2 = np.zeros((output_size, 1))
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def forwardPropagation(X, params):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(params['W2'], A1)+params['b2']
    y = sigmoid(Z2)  
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}
def cost(predict, actual):
    m = actual.shape[1]
    cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
    return np.squeeze(cost__)
def backPropagation(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(params['W2']), dy) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
def updateParameters(gradients, params, learning_rate = 1.2):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
def fit(X, Y, learning_rate, hidden_size, number_of_iterations = 5000):
    params = setParameters(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forwardPropagation(X, params)
        costit = cost(y, Y)
        gradients = backPropagation(X, Y, params, cache)
        params = updateParameters(gradients, params, learning_rate)
        cost_.append(costit)
    return params, cost_

import os
import cv2
import numpy as np

def load_images_from_directory(directory):
    images = []
    labels = []
    class_folders = sorted(os.listdir(directory))
    for class_label, class_folder in enumerate(class_folders):
        class_path = os.path.join(directory, class_folder)
        for filename in os.listdir(class_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions as needed
                image_path = os.path.join(class_path, filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (32, 32))  # Resize images to 32x32 pixels
                images.append(image)
                labels.append(class_label)

    return np.array(images), np.array(labels)

# Define paths to your training and testing directories
train_directory = './train'
test_directory = './test'

# Load and preprocess the training dataset
train_x_orig, train_y = load_images_from_directory(train_directory)

# Load and preprocess the testing dataset
test_x_orig, test_y = load_images_from_directory(test_directory)

# Reshape and normalize the image data
train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.
train_y = train_y.reshape(1, -1)
test_y = test_y.reshape(1, -1)
# def predict(test_x, test_y, parameters,threshold):

#     """
#     Predict test data
#     test_x -- test data
#     test_y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    
#     Returns:
#     accuracy -- accuracy of your model
    
#     """
#     predictions = np.zeros((1,test_x.shape[1]))
#     pred=forwardPropagation(test_x, parameters)
    
#     for i in range(0, pred.shape[1]):
#         predictions[0,i] = (pred[0,i] > threshold)
    
#     accuracy = np.sum((predictions == test_y)/test_x.shape[1])
#     return predictions, accuracy
def predict(X, params):
    """
    Make predictions using a trained neural network.

    Arguments:
    X -- input data
    params -- parameters of the trained neural network

    Returns:
    predictions -- predicted labels
    """
    _, cache = forwardPropagation(X, params)
    predictions = (cache['y'] > 0.5).astype(int)
    
    return predictions
import sklearn.datasets
X, Y = train_x,train_y
# X, Y = X.T, Y.reshape(1, Y.shape[0])
# print(X.shape)
# print(Y.shape)
params, cost_ = fit(X, Y, 0.00008, 5, 500)
test_predictions = predict(test_x, params)
print("Test Accuracy: " + str(test_predictions))
# Evaluate accuracy
test_accuracy = np.mean((test_predictions == test_y).astype(int)) * 100
print(f"\nAccuracy on Test Set: {test_accuracy:.2f}%")
import matplotlib.pyplot as plt
plt.plot(cost_)