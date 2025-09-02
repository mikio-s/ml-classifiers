import numpy as np
from sklearn.linear_model import Perceptron
import time

# Interactive plots
%matplotlib ipympl
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

def perceptron_update(x, y, w):
    """
    function w=perceptron_update(x,y,w);

    Updates the perceptron weight vector w using x and y
    Input:
    x : input vector of d dimensions (d)
    y : corresponding label (-1 or +1)
    w : weight vector of d dimensions

    Output:
    w : weight vector after updating (d)
    """
    w = w + y*x
    return w


# Little test
y = -1
x = np.random.randint(low=-2, high=2, size=10)
w = np.random.randint(low=0, high=2, size=10) * 2 - 1
print(f"y     = {y}")
print(f"x     = {x}")
print(f"w_old = {w}")

w_new = perceptron_update(x, y, w)
print(f"w_new = {w_new}")

def perceptron_update_test1():
    x = np.array([0, 1])
    y = -1
    w = np.array([1, 1])
    w1 = perceptron_update(x, y, w)
    return (w1.reshape(-1,) == np.array([1, 0])).all()


def perceptron_update_test2():
    x = np.random.rand(25)
    y = 1
    w = np.zeros(25)
    w1 = perceptron_update(x, y, w)
    return np.linalg.norm(w1 - x) < 1e-8


def perceptron_update_test3():
    x = np.random.rand(25)
    y = -1
    w = np.zeros(25)
    w1 = perceptron_update(x, y, w)
    return np.linalg.norm(w1 + x) < 1e-8


runtest(perceptron_update_test1, "perceptron_update_test1")
runtest(perceptron_update_test2, "perceptron_update_test2")
runtest(perceptron_update_test3, "perceptron_update_test3")

def perceptron(xs, ys):
    """
    function w=perceptron(xs,ys);

    Returns the weight vector learned by the Perceptron classifier.

    Input:
        xs: n input vectors of d dimensions (nxd matrix)
        ys: n labels (-1 or +1)

    Output:
        w: weight vector (d)
        b: bias term
    """
    n, d = xs.shape  # So we have n input vectors with d dimensions
    w = np.zeros(d)
    b = 0.0

    ite = 0
    while ite < 100:
        np.random.permutation(n)
        for i in range (0,n):
            if (ys[i] * (np.dot(w.T, xs[i]) + b)) <= 0:
                w = perceptron_update(xs[i], ys[i], w)
                b += ys[i]
            ite += 1
            
    #print(w, b)

    return (w, b)def perceptron_test1():
    N = 100
    d = 10
    x = np.random.rand(N, d)
    w = np.random.rand(1, d)
    y = np.sign(w.dot(x.T))[0]
    w, b = perceptron(x, y)
    preds = classify_linear_grader(x, w, b)
    return np.array_equal(preds.reshape(-1,), y.reshape(-1,))


def perceptron_test2():
    x = np.array(
        [
            [-0.70072, -1.15826],
            [-2.23769, -1.42917],
            [-1.28357, -3.52909],
            [-3.27927, -1.47949],
            [-1.98508, -0.65195],
            [-1.40251, -1.27096],
            [-3.35145, -0.50274],
            [-1.37491, -3.74950],
            [-3.44509, -2.82399],
            [-0.99489, -1.90591],
            [0.63155, 1.83584],
            [2.41051, 1.13768],
            [-0.19401, 0.62158],
            [2.08617, 4.41117],
            [2.20720, 1.24066],
            [0.32384, 3.39487],
            [1.44111, 1.48273],
            [0.59591, 0.87830],
            [2.96363, 3.00412],
            [1.70080, 1.80916],
        ]
    )
    y = np.array([1] * 10 + [-1] * 10)
    w, b = perceptron(x, y)
    preds = classify_linear_grader(x, w, b)
    return np.array_equal(preds.reshape(-1,), y.reshape(-1,))


runtest(perceptron_test1, "perceptron_test1")
runtest(perceptron_test2, "perceptron_test2")

# Number of observations and features
N, d = 500, 2

# Create random dataset
np.random.seed(1)
x = np.random.rand(N, d)
x -= np.mean(x)  # Subtracting mean puts features in range [-0.5, 0.5]

# Generate w from which labels are computed
w_true = np.random.rand(1, d)
w_true -= np.mean(w_true)  # Only rotates weight vector in 2D space
y_true = np.sign(w_true.dot(x.T))[0]

# Make the separation line impossible to compute so that 
# both w_true and w_pred can be plotted distinctly.
y_true[0] = -1 * y_true[0]

# Compute predictions
w_pred, b = perceptron(x, y_true)
y_pred = classify_linear_grader(x, w_pred, b).astype(int)

plot_perceptron_2D(x, y_true, w_true, y_pred, w_pred, b)

# Number of input vectors
N = 100

# Generate random (linarly separable) data
xs = np.random.rand(N, 2) * 10 - 5

# Defining random hyperplane
w0 = np.random.rand(2)
b0 = np.random.rand() * 2 - 1

# Assigning labels (+1 or -1) depending on what side of the plane they lie on
ys = np.sign(xs.dot(w0) + b0)

# Call perceptron to find w from data
w, b = perceptron(xs.copy(), ys.copy())

# Test if all points are classified correctly
assert all(np.sign(ys * (xs.dot(w) + b)) == 1.0)  # Should be +1.0 for every input
print("Looks like you passed the perceptron test!")

# We can make a pretty visualization
visboundary(w, b, xs, ys)

# Initialize empty array of width 3 to store two-dimensional training points
# and a third dimension that has a constant value of 1 (to incorporate bias).
xy_data = np.empty((0, 3))

# Initialize empty array of width 1 to store training labels
label_data = np.empty((0, 1))


def onclick(event):
    """
    Assume the last dimension of each training point is 1,
    so the last dimension of w represents the bias term.
    """
    global xy_data, label_data

    # Shift+click to add a negative point.
    # Click to add a positive point.
    if event.key == "shift":
        label = -1
    else:
        label = 1

    # Create position vector for new point with
    # x-coord, y-coord, and constant value of 1.
    pos = np.array([[event.xdata, event.ydata, 1]])

    # Add new point and label to training data
    xy_data = np.vstack((xy_data, pos))
    label_data = np.append(label_data, label)

    # Get class values from training labels
    classvals = np.unique(label_data)

    # Plot configuration
    plt.cla()                  # Clear axes
    plt.xlim((0, 1))           # x-axis limits
    plt.ylim((0, 1))           # y-axis limits
    labels = ["$-1$", "$+1$"]  # Labels for training points
    symbols = ["x", "o"]       # "x" for -1 and "o" for +1 (training)
    colors = ["red", "blue"]   # Red for -1 and blue for +1 (predictions)

    if len(classvals) == 1 and 1 in classvals:
        # Plot points with "o" markers because only +1 points have been added
        for idx, c in enumerate(classvals):
            plt.scatter(
                xy_data[label_data == c, 0],  # x-coordinate of training point
                xy_data[label_data == c, 1],  # y-coordinate of training point
                label="$+1$",                 # Label of training point
                marker="o",                   # Marker of training point
                color="blue",                 # Color of training point
            )
    else:
        # Plot training data: "x" for -1 and "o" for +1
        for idx, c in enumerate(classvals):
            plt.scatter(
                xy_data[label_data == c, 0],  # x-coordinate of training point
                xy_data[label_data == c, 1],  # y-coordinate of training point
                label=labels[idx],            # Label of training point (-1 or +1)         
                marker=symbols[idx],          # Marker of training point ("x" or "o")
                color=colors[idx],            # Color of training point
            )
    
    # Call perceptron function to get trained weights and bias
    w, b = perceptron(xy_data, label_data)

    # Plot decision boundary using trained weights and bias
    x = np.linspace(start=0, stop=1, num=1000)
    y = -(w[0] * x + b) / w[1]
    plt.plot(x, y, color="black", linestyle="--")
    
    plt.title("Click: Positive Point, Shift+Click: Negative Point")
    plt.legend(loc="upper left")
    plt.show()


# Plot interactive demo
print("Please keep in mind:")
print("1. You must run (or rerun) this cell right before interacting with the plot.")
print("2. Start the interactive demo by clicking the grid to add a positive point.")
print("3. Click to add a positive point or shift+click to add a negative point.")
print("4. You may notice a slight delay when adding points to the visualization.")
fig = plt.figure()
plt.title("Start by Clicking the Grid to Add a Positive Point")
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect("button_press_event", onclick)

def classify_linear(xs, w, b=None):
    """
    function preds=classify_linear(xs,w,b)

    Make predictions with a linear classifier
    
    Input:
        xs: n input vectors of d dimensions (nxd) [could also be a single vector of d dimensions]
        w: weight vector of dimensionality d
        b: bias (scalar)

    Output:
        preds: predictions (1xn)
    """
    w = w.flatten()
    predictions = np.zeros(xs.shape[0])

    # YOUR CODE HERE
    if b is None: 
        b=0.0
    else: b
    predictions = np.sign(np.dot(xs, w) + b)
    
    # END OF YOUR CODE
    
    return predictions

def linear_test1():
    """
    Check that predictions are only -1 or 1
    """
    xs = np.random.rand(50000, 20) - 0.5  # Draw random data
    xs = np.hstack([xs, np.ones((50000, 1))])
    w0 = np.random.rand(20 + 1)
    w0[-1] = -0.1                         # Assign bias of -0.1
    ys = classify_linear(xs, w0)
    uniquepredictions = np.unique(ys)
    return set(uniquepredictions) == {-1, 1}


def linear_test2():
    """
    Check that all predictions are correct for linearly separable data
    """
    xs = np.random.rand(1000, 2) - 0.5  # Draw random data
    xs = np.hstack([xs, np.ones((1000, 1))])

    # Define a random hyperplane with bias -0.1
    w0 = np.array([0.5, -0.3, -0.1])
    
    # Assign labels according to this hyperplane (so you know it is linearly separable)
    ys = np.sign(xs.dot(w0))

    # Original hyperplane (w0, b0) should classify all ones correctly
    result = all(np.sign(ys * classify_linear(xs, w0)) == 1.0)
    
    return result


runtest(linear_test1, "linear_test1")
runtest(linear_test2, "linear_test2")
