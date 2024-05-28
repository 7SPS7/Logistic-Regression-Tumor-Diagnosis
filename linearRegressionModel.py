import numpy as np
import matplotlib.pyplot as plt

# Provided code for data and logistic regression implementation

X_train_given = np.array([
    [12, 1.64], [12, 4.35], [12, 2.18], [12, 3.79], [12, 9.21], [12, 8.05], [17, 8.34], [17, 3.22], [17, 9.57],
    [17, 10.00], [21, 5.66], [21, 7.82], [21, 3.98], [21, 3.25], [23, 2.11], [23, 6.07], [23, 10.51], [23, 5.70],
    [27, 9.31], [27, 5.66], [27, 3.32], [27, 2.74], [34, 6.50], [34, 3.65], [34, 10.59], [34, 6.86], [34, 2.57],
    [34, 6.66], [34, 4.38], [34, 10.50], [34, 9.58], [34, 5.14], [38, 9.11], [38, 7.66], [38, 3.87], [38, 6.54],
    [38, 6.02], [38, 6.63], [38, 10.27], [38, 9.65], [39, 4.24], [39, 5.39], [39, 4.58], [39, 8.26], [42, 7.28],
    [42, 5.39], [42, 8.74], [42, 2.99], [48, 5.97], [48, 7.86], [48, 6.55], [48, 3.58], [50, 9.49], [50, 5.68],
    [50, 2.14], [50, 7.40], [53, 2.95], [53, 3.32], [53, 7.23], [53, 6.23], [55, 6.14], [55, 2.27], [55, 4.72],
    [55, 10.38], [61, 5.18], [61, 7.87], [61, 7.79], [61, 9.22], [62, 9.87], [62, 2.07], [62, 10.94], [62, 2.35],
    [67, 7.33], [67, 5.86], [67, 5.31], [67, 3.46], [68, 2.47], [68, 4.90], [68, 5.38], [68, 4.08], [69, 2.89],
    [69, 9.50], [69, 10.60], [69, 5.01], [70, 7.18], [70, 7.72], [70, 2.38], [70, 8.14], [76, 5.46], [76, 10.77],
    [76, 7.76], [76, 7.19], [76, 2.13], [76, 5.51], [76, 5.75], [76, 3.32], [80, 10.82], [80, 3.74], [80, 10.18],
    [80, 7.19],
])

Y_train = np.array([
    0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1.,
    0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1.,
    0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0.,
    0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
])

# Logistic Regression Implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # to prevent log(0) case
    cost = (1/m) * ((-y).T.dot(np.log(h + epsilon)) - (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# Data preprocessing

# Round off each data point
X_train_rounded = np.round(X_train_given)

# Scale down the size of the second column elements by 10
X_train_scaled = X_train_rounded.copy()
X_train_scaled[:, 1] /= 10

num_samples = 100
age = X_train_rounded
tumor_size = Y_train

# Combine age and tumor_size into feature matrix X
X = np.column_stack((age, tumor_size))

# Generate labels (0: benign, 1: malignant)
y = np.random.randint(0, 2, num_samples)

# Plot the data
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Benign')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Malignant')

plt.xlabel('Age')
plt.ylabel('Tumor Size')
plt.title('Synthetic Tumor Diagnosis Dataset (100 data points)')
plt.legend()
plt.show()

# Add intercept term to X
X_with_intercept = np.column_stack((np.ones((num_samples, 1)), X))

# Initialize theta parameters
initial_theta = np.zeros(X_with_intercept.shape[1])

# Set hyperparameters
alpha = 0.05
num_iters = 50

# Perform gradient descent
theta, cost_history = gradient_descent(X_with_intercept, y, initial_theta, alpha, num_iters)

# Plot cost history to check convergence
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

# Display final parameters
print("Final Parameters:")
print("Theta:", theta)

# Input from user
age_input = float(input("Enter the age of the patient: "))
tumor_size_input = float(input("Enter the size of the tumor: "))

# Scale down tumor size
tumor_size_input_scaled = tumor_size_input / 10

# Add intercept term to input features
input_features = np.array([[1, 1, age_input, tumor_size_input_scaled]])

# Predict using the trained logistic regression model
prediction_probability = sigmoid(input_features.dot(theta))
predicted_label = "Malignant" if prediction_probability >= 0.5 else "Benign"

# Display prediction
print("Based on the input data:")
print("Age:", age_input)
print("Tumor Size:", tumor_size_input)
print("Prediction:", predicted_label)
print("Probability of being malignant:", prediction_probability)
