import numpy as np
import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LogisticRegression

import numpy as np

# Placeholder values, replace them with your actual data
age = np.array([12, 17, 21, 23, 27, 34, 38, 39, 42, 48, 50, 53, 55, 61, 62, 67, 68, 69, 70, 76, 80, 90])
tumor_size = np.array([1.64, 4.35, 2.18, 3.79, 9.21, 8.05, 8.34, 3.22, 9.57, 10.00, 5.66, 7.82, 3.98, 3.25, 2.11, 6.07, 10.51, 5.70, 9.31, 5.66, 3.32, 2.74])
num_samples = len(age)
y = np.random.randint(0, 2, num_samples)  # Generate random labels for demonstration purposes


class TumorDiagnosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Diagnosis")

        ttk.Label(root, text="Age:").grid(row=0, column=0, padx=5, pady=5)
        self.age_entry = ttk.Entry(root)
        self.age_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(root, text="Tumor Size:").grid(row=1, column=0, padx=5, pady=5)
        self.size_entry = ttk.Entry(root)
        self.size_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(root, text="Diagnose", command=self.diagnose).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def diagnose(self):
        age = float(self.age_entry.get())
        tumor_size = float(self.size_entry.get())

        # Prepare input for prediction
        input_data = np.array([[age, tumor_size]])

        # Use logistic regression model for prediction
        prediction = self.predict(input_data)

        # Display the diagnosis result
        if prediction == 0:
            result = "Benign tumor"
        else:
            result = "Malignant tumor"

        result_label = ttk.Label(self.root, text=result)
        result_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

    def predict(self, input_data):
        # Load the logistic regression model
        model = LogisticRegression()
        model.fit(X_with_intercept[:, 1:], y)  # Fit the model with training data

        # Predict using the model
        prediction = model.predict(input_data)
        return prediction


# Given code for data and logistic regression model
# (Assuming theta and X_with_intercept are defined)

# Logistic Regression Implementation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # to prevent log(0) case
    cost = (1 / m) * ((-y).T.dot(np.log(h + epsilon)) - (1 - y).T.dot(np.log(1 - h + epsilon)))
    return cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


# Combine age and tumor_size into feature matrix X
X = np.column_stack((age, tumor_size))

# Combine age and tumor_size into feature matrix X
X = np.column_stack((age, tumor_size))

# Add intercept term to X
X_with_intercept = np.column_stack((np.ones((num_samples, 1)), X))

# Initialize theta parameters
initial_theta = np.zeros(X_with_intercept.shape[1])

# Set hyperparameters
alpha = 0.05
num_iters = 50

# Perform gradient descent
theta, cost_history = gradient_descent(X_with_intercept, y, initial_theta, alpha, num_iters)

# Create and run the Tkinter application
if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDiagnosisApp(root)
    root.mainloop()
