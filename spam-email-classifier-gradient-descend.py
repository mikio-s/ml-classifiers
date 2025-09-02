import numpy as np
import pandas as pd

# Non-interactive plots
%matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

np.random.seed(12)
n_samples = 500
covariance = [[1, 0.25], [0.25, 1]]

class_one = np.random.multivariate_normal(mean=[0, 5], cov=covariance, size=n_samples)
class_one_labels = np.ones(n_samples)

class_two = np.random.multivariate_normal(mean=[5, 10], cov=covariance, size=n_samples)
class_two_labels = -np.ones(n_samples)

features = np.vstack((class_one, class_two))
labels = np.hstack((class_one_labels, class_two_labels))

print("Features shape:", features.shape)
print(features.round(3))

df = pd.DataFrame(
    np.vstack([labels, features[:, 0], features[:, 1]]).T,
    columns=["label", "feature_1", "feature_2"],
)
display(df.groupby("label").describe())

plt.figure()

plt.title("Artificially Generated 2D Data With Binary Labels")
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)

plt.scatter(
    features[labels == 1, 0],  # x-coordinate
    features[labels == 1, 1],  # y-coordinate
    label="Class 1 ($+1$)",
    color="blue",
    alpha=0.5,
)
plt.scatter(
    features[labels != 1, 0],  # x-coordinate
    features[labels != 1, 1],  # y-coordinate
    label="Class 2 ($-1$)",
    color="red",
    alpha=0.5,
)

plt.legend(loc="upper left")
plt.show()

def sigmoid(z):
    """
    Calculates the sigmoid of z.

    Input:
        z: scalar or array of dimension n

    Output:
        scalar or array of dimension n
    """
    sgmd = 1 / (1 + np.exp(-z))

    return sgmd


z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure()
plt.plot(z, y)
plt.title(r"Graph of the Sigmoid Function $\sigma(z)$")
plt.grid()
plt.show()

def sigmoid_test1():
    """
    Check that your sigmoid output has the correct shape
    """
    h = np.random.rand(10)              # Input is a random 10-dimensional array
    return sigmoid(h).shape == h.shape  # Output should be a 10-dimensional array


def sigmoid_test2():
    """
    Check your sigmoid outputs against our sigmoid outputs
    """
    np.random.seed(1)          # Set a random seed for consistency
    h = np.random.rand(10)     # Input is a random 10-dimensional array
    sgmd1 = sigmoid(h)         # Compute sigmoids with your function
    sgmd2 = sigmoid_grader(h)  # Compute sigmoids with our function
    diff_sgmds = np.linalg.norm(sgmd1 - sgmd2)
    return diff_sgmds < 1e-5   # Difference should be small


def sigmoid_test3():
    """
    Another check that your sigmoid output is close to ours
    """
    np.random.seed(2)          # Set a random seed for consistency
    x = np.random.rand(1)      # Input is a random scalar
    sgmd1 = sigmoid(x)         # Compute sigmoid with your function
    sgmd2 = sigmoid_grader(x)  # Compute sigmoid with our function
    diff_sgmds = np.linalg.norm(sgmd1 - sgmd2)
    return diff_sgmds < 1e-5   # Difference should be small


def sigmoid_test4():
    """
    Check your sigmoid outputs against known sigmoid outputs
    """
    # Three inputs: very negative, very positive, and zero
    x = np.array([-1e10, 1e10, 0])  

    # Compute the output with your sigmoid function
    sgmd = sigmoid(x)

    # The first two sigmoid outputs should be very close to 0 and 1,
    # and the last sigmoid output should be exactly 0.5.
    truth = np.array([0, 1, 0.5])
    diff_sgmd_truth = np.linalg.norm(sgmd - truth)
    
    return diff_sgmd_truth < 1e-8  # Difference should be very small


runtest(sigmoid_test1, "sigmoid_test1")
runtest(sigmoid_test2, "sigmoid_test2")
runtest(sigmoid_test3, "sigmoid_test3")
runtest(sigmoid_test4, "sigmoid_test4")

def y_pred(X, w, b=0):
    """
    Calculates the probability of the positive class.

    Input:
        X: data matrix of shape nxd
        w: d-dimensional vector
        b: scalar (optional, default is 0)

    Output:
        n-dimensional vector
    """
    prob = sigmoid(X@w + b)

    return prob


n, d = 5, 3
X = np.random.randint(0, 3, (n, d))  # Contains multiple observations as rows
w = np.random.randint(-3, 3, d)      # This will be our trained vector of coefficients
b = 1                                # This will be our trained offset (or bias) scalar
pY = y_pred(X, w, b).round(2)
print(f"X = \n{X}")
print(f"w = {w}, b = {b}")
print(f"Vector of probabilities of class +1 (conditional on x and w) = {pY}")
print(f"Vector of probabilities of class -1 (conditional on x and w) = {1-pY}")

def y_pred_test1():
    """
    Check that your y_pred output has the correct shape
    """
    # Generate n random vectors with d dimensions
    n = 20
    d = 5
    X = np.random.rand(n, d)

    # Define a random weight vector
    w = np.random.rand(d)

    # Compute probabilities P(y=1|X;w,b=0) using your y_pred function
    probs = y_pred(X, w, 0)

    # Check if the output of y_pred has the correct shape
    return probs.shape == (n,)


def y_pred_test2():
    """
    Check that your y_pred output has the correct range
    """
    # Generate n random vectors with d dimensions
    n = 20
    d = 5
    X = np.random.rand(n, d)

    # Define a random weight vector
    w = np.random.rand(d)

    # Compute probabilities P(y=1|X;w,b=0) using your y_pred function
    probs = y_pred(X, w, 0)

    # Check if all outputs are >=0 and <=1
    return all(probs >= 0) and all(probs <= 1)


def y_pred_test3():
    """
    Check that your y_pred outputs sum to one
    """
    # Generate n random vectors with d dimensions
    n = 20
    d = 5
    X = np.random.rand(n, d)

    # Define a random weight vector
    w = np.random.rand(d)

    # Compute probabilities P(y=1|X;w,b=0) using your y_pred function
    probs1 = y_pred(X, w, 0)

    # Compute probabilities P(y=-1|X;w,b=0) = P(y=1|X;-w,b=0) using your y_pred function
    probs2 = y_pred(X, -w, 0)

    # Check if P(y=1|X;w,b=0) + P(y=-1|X;w,b=0) = 1
    return np.linalg.norm(probs1 + probs2 - 1) < 1e-08


def y_pred_test4():
    """
    Check that your y_pred function matches true values
    """
    # Set a random seed for consistency
    np.random.seed(1)
    
    # Define random input
    X = np.random.rand(25, 4)

    # Weight vector with all weight on the first feature
    w = np.array([1, 0, 0, 0])

    # Compute P(y=1|X;w,b=0)
    prob = y_pred(X, w, 0)

    # Ground truth should be the sigmoid of the first feature
    truth = sigmoid(X[:, 0])

    # Check if predictions match the ground truth
    return np.linalg.norm(prob - truth) < 1e-08


def y_pred_test5():
    """
    Another check that your y_pred function matches true values
    """
    # Define 3 inputs (2D)
    X = np.array(
        [
            [0.61793598, 0.09367891],
            [0.79447745, 0.98605996],
            [0.53679997, 0.4253885],
        ]
    )
    
    # Define weight vector
    w = np.array([0.9822789, 0.16017851])

    # Compute probabilities P(y=1|X;w,b=3)
    prob = y_pred(X, w, 3)

    # This is the ground truth
    truth = np.array([0.97396645, 0.98089179, 0.97328431])

    # Check if predictions match the ground truth
    return np.linalg.norm(prob - truth) < 1e-08


runtest(y_pred_test1, "y_pred_test1")
runtest(y_pred_test2, "y_pred_test2")
runtest(y_pred_test3, "y_pred_test3")
runtest(y_pred_test4, "y_pred_test4")
runtest(y_pred_test5, "y_pred_test5")

def log_loss(X, y, w, b=0):
    """
    Calculates the negative log-likelihood for dataset (X, y)
    using the weight vector w and bias b.

    Input:
        X: data matrix of shape nxd
        y: n-dimensional vector of labels (+1 or -1)
        w: d-dimensional vector
        b: scalar (optional, default is 0)

    Output:
        scalar
    """
    assert np.sum(np.abs(y)) == len(y)  # Check all labels are either +1 or -1

    z = np.dot(X, w) + b
    nll = -np.sum(np.log(sigmoid(y * z)))
    #print(nll)

    return nll


def log_loss_test1():
    """
    Check that the output of your log_loss function is a scalar
    """
    X = np.random.rand(25, 5)  # Generate n random vectors with d dimensions
    w = np.random.rand(5)      # Define a random weight vector
    b = np.random.rand(1)      # Define a random bias term
    y = np.ones(25)            # Set all labels to +1 (ground truth)
    ll = log_loss(X, y, w, b)  # Compute log loss between y and y_pred
    return np.isscalar(ll)


def log_loss_test2():
    """
    Check that your log_loss function yields the correct output.
    Note: If the labels are all one, the output of your log_loss
    function should be the (negative) sum of log of y_pred.
    """
    X = np.random.rand(25, 5)   # Generate n random vectors with d dimensions
    w = np.random.rand(5)       # Define a random weight vector
    b = np.random.rand(1)       # Define a random bias term
    y = np.ones(25)             # Set all labels to +1 (ground truth)
    ll1 = log_loss(X, y, w, b)  # Compute log loss between y and y_pred
    ll2 = -np.sum(np.log(y_pred(X, w, b)))    # Negative sum of log of y_pred
    return np.linalg.norm(ll1 - ll2) < 1e-05  # Difference should be small


def log_loss_test3():
    """
    Another check that your log_loss function yields the correct output.
    Note: If the labels are all negative one, the output of your log_loss
    function should be the (negative) sum of log of 1-y_pred.
    """
    X = np.random.rand(25, 5)   # Generate n random vectors with d dimensions
    w = np.random.rand(5)       # Define a random weight vector
    b = np.random.rand(1)       # Define a random bias term
    y = -np.ones(25)            # Set all labels to -1 (ground truth)
    ll1 = log_loss(X, y, w, b)  # Compute log loss between y and y_pred
    ll2 = -np.sum(np.log(1 - y_pred(X, w, b)))  # Negative sum of log of 1-y_pred
    return np.linalg.norm(ll1 - ll2) < 1e-05    # Difference should be small


def log_loss_test4():
    """
    Another check that your log_loss function yields the correct output
    """
    X = np.random.rand(20, 5)               # Generate n random vectors with d dimensions
    w = np.array([0, 0, 0, 0, 0])           # Define a weight vector with all zeros
    y = (np.random.rand(20) > 0.5) * 2 - 1  # Define n random labels +1 or -1 (ground truth)
    ll = log_loss(X, y, w, 0)               # Compute log loss between y and y_pred
    
    # The log-likelihood for each of the 20 examples should be exactly 0.5
    return np.linalg.norm(ll + 20 * np.log(0.5)) < 1e-08


def log_loss_test5():
    """
    Check the output of your log_loss against the output of ours
    """
    np.random.seed(1)                         # Set a random seed for consistency
    X = np.random.rand(500, 15)               # Generate n random vectors with d dimensions
    w = np.random.rand(15)                    # Define a random weight vector
    b = np.random.rand(1)                     # Define a random bias term
    y = (np.random.rand(500) > 0.5) * 2 - 1   # Define n random labels +1 or -1 (ground truth)
    ll1 = log_loss(X, y, w, b)                # Compute your log loss between y and y_pred
    ll2 = log_loss_grader(X, y, w, b)         # Compute our log loss between y and y_pred
    return np.linalg.norm(ll1 - ll2) < 1e-05  # Difference should be small


runtest(log_loss_test1, "log_loss_test1")
runtest(log_loss_test2, "log_loss_test2")
runtest(log_loss_test3, "log_loss_test3")
runtest(log_loss_test4, "log_loss_test4")
runtest(log_loss_test5, "log_loss_test5")


#Implementing Gradien Descend

def gradient(X, y, w, b):
    """
    Calculates the gradients of negative log-likelihood (NLL)  
    with respect to the weight vector w and the bias term b.
    Returns the tuple (w_grad, bgrad).

    Input:
        X: Data matrix of shape (n, d)
        y: n-dimensional vector of labels (+1 or -1)
        w: d-dimensional weight vector
        b: Scalar bias term

    Output:
        wgrad: d-dimensional vector (gradient vector of w)
        bgrad: Scalar (gradient of b)
    """
    n, d = X.shape
    wgrad = np.zeros(d)
    bgrad = 0.0

    z = np.dot(X, w) + b
    d = sigmoid_grader(-y * z) * y
    wgrad = -np.dot(X.T, d)
    bgrad = -np.sum(d)

    return wgrad, bgrad

def grad_test1():
    """
    Check that your gradient has the correct shape
    """
    X = np.random.rand(25, 5)               # Generate random data matrix
    w = np.random.rand(5)                   # Generate random weight vector
    b = np.random.rand(1)                   # Generate random bias term
    y = (np.random.rand(25) > 0.5) * 2 - 1  # Generate labels that are all +1
    wgrad, bgrad = gradient(X, y, w, b)     # Compute your gradient
    return wgrad.shape == w.shape and np.isscalar(bgrad)


def grad_test2():
    """
    Check your gradient against our gradient
    """
    np.random.seed(1)                       # Random seed for consistency
    X = np.random.rand(25, 5)               # Generate random data matrix
    w = np.random.rand(5)                   # Generate random weight vector
    b = np.random.rand(1)                   # Generate random bias term
    y = (np.random.rand(25) > 0.5) * 2 - 1  # Generate labels that are all +1
    wgrad, bgrad = gradient(X, y, w, b)           # Compute your gradient
    wgrad2, bgrad2 = gradient_grader(X, y, w, b)  # Compute our gradient

    # Check whether the difference in gradients is small
    w_diff_small = np.linalg.norm(wgrad - wgrad2) < 1e-06
    b_diff_small = np.linalg.norm(bgrad - bgrad2) < 1e-06
    
    return w_diff_small and b_diff_small


def grad_test3():
    """
    Check your log loss against our log loss
    """
    np.random.seed(2)                       # Random seed for consistency
    X = np.random.rand(25, 5)               # Generate random data matrix
    w = np.random.rand(5)                   # Generate random weight vector
    b = np.random.rand(1)                   # Generate random bias term
    y = (np.random.rand(25) > 0.5) * 2 - 1  # Generate labels that are all +1

    w_s = np.random.rand(5) * 1e-05         # Small random step
    b_s = np.random.rand(1) * 1e-05         # Small random step

    # Compute our log loss
    ll = log_loss_grader(X, y, w, b)
    
    # Compute our log loss after taking small random step
    ll1 = log_loss_grader(X, y, w + w_s, b + b_s)

    # Compute your gradient to approximate log loss using Taylor's expansion
    wgrad, bgrad = gradient(X, y, w, b)   

    # Use Taylor's expansion to approximate your log loss after small random step
    ll2 = ll + wgrad @ w_s + bgrad * b_s

    # Check whether the difference in log loss is small
    ll_diff_small = np.linalg.norm(ll1 - ll2) < 1e-05
    
    return ll_diff_small


def grad_test4():
    """
    Check whether your logistic regression losses are close to our losses
    """
    # Get your losses (i.e., losses1)
    w1, b1, losses1 = logistic_regression_grader(features, labels, 1000, 1e-03, gradient)

    # Get our losses (i.e., losses2)
    w2, b2, losses2 = logistic_regression_grader(features, labels, 1000, 1e-03)

    # Check whether the difference in losses is small
    losses_diff_small = np.abs(losses1[-1] - losses2[-1]) < 0.1
    
    return losses_diff_small


runtest(grad_test1, "grad_test1")
runtest(grad_test2, "grad_test2")
runtest(grad_test3, "grad_test3")
runtest(grad_test4, "grad_test4")


def logistic_regression(X, y, max_iter, alpha):
    """
    Trains the logistic regression classifier on data X and labels y
    using gradient descent for max_iter iterations with learning rate alpha.
    Returns the weight vector, bias term, and losses at each iteration AFTER
    updating the weight vector and bias.

    Input:
        X: Data matrix of shape nxd
        y: n-dimensional vector of data labels (+1 or -1)
        max_iter: Number of iterations of gradient descent to perform
        alpha: Learning rate for each gradient descent step

    Output:
        w, b, losses
        w: d-dimensional weight vector
        b: Scalar bias term
        losses: max_iter-dimensional vector containing negative log-likelihood
                values AFTER a gradient descent in each iteration
    """
    d = X.shape[1]
    w = np.zeros(d)
    b = 0.0
    losses = np.zeros(max_iter)

    for step in range(max_iter):
        wgrad, bgrad = gradient(X, y, w, b)
        w = w - alpha * wgrad
        b = b - alpha * bgrad
        losses[step] = log_loss_grader(X, y, w, b)
    
    return w, b, losses

def logistic_regression_test1():
    """
    Check that your trained weights and bias are close to ours
    """
    XUnit = np.array(
        [
            [-1, 1],
            [-1, 0],
            [0, -1],
            [-1, 2],
            [1, -2],
            [1, -1],
            [1, 0],
            [0, 1],
            [1, -2],
            [-1, 2],
        ]
    )
    YUnit = np.hstack((np.ones(5), -np.ones(5)))

    # Your trained weights and bias
    w1, b1, _ = logistic_regression(XUnit, YUnit, 30000, 5e-5)

    # Our trained weights and bias
    w2, b2, _ = logistic_regression_grader(XUnit, YUnit, 30000, 5e-5)

    # Check that the differences are small
    w_diff_small = np.linalg.norm(w1 - w2) < 1e-5
    b_diff_small = np.linalg.norm(b1 - b2) < 1e-5
    
    return w_diff_small and b_diff_small


def logistic_regression_test2():
    """
    Another check that your trained weights and bias are close to ours
    """
    np.random.seed(1)  # Set random seed for consistency
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 300
    alpha = 1e-5

    # Your trained weights and bias
    w1, b1, _ = logistic_regression(X, Y, max_iter, alpha)

    # Our trained weights and bias
    w2, b2, _ = logistic_regression_grader(X, Y, max_iter, alpha)

    # Check that the differences are small
    w_diff_small = np.linalg.norm(w1 - w2) < 1e-5
    b_diff_small = np.linalg.norm(b1 - b2) < 1e-5
    
    return w_diff_small and b_diff_small


def logistic_regression_test3():
    """
    Check that your final loss matches our final loss
    """
    np.random.seed(2)  # Set random seed for consistency
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 30
    alpha = 1e-5

    # Compute your final loss
    w, b, losses = logistic_regression(X, Y, max_iter, alpha)
    your_loss = losses[-1]

    # Compute our final loss (with your trained weights and bias)
    our_loss = log_loss_grader(X, Y, w, b)

    # Check that the difference is small
    loss_diff_small = np.abs(your_loss - our_loss) < 1e-09
    
    return loss_diff_small


def logistic_regression_test4():
    """
    Check that your losses decrease
    """
    np.random.seed(3)  # Set random seed for consistency
    X = np.vstack((np.random.randn(50, 5), np.random.randn(50, 5) + 2))
    Y = np.hstack((np.ones(50), -np.ones(50)))
    max_iter = 30
    alpha = 1e-5

    # Get your first loss and last loss
    _, _, losses = logistic_regression(X, Y, max_iter, alpha)
    first_loss = losses[0]
    last_loss = losses[-1]
    
    return last_loss < first_loss


runtest(logistic_regression_test1, "logistic_regression_test1")
runtest(logistic_regression_test2, "logistic_regression_test2")
runtest(logistic_regression_test3, "logistic_regression_test3")
runtest(logistic_regression_test4, "logistic_regression_test4")

max_iter = 10000  # Max number of iterations
alpha = 1e-4      # Learning rate

# Train the logistic regression model
final_w, final_b, losses = logistic_regression(features, labels, max_iter, alpha)

plt.figure()
plt.title("Training Loss for Logistic Regression")
plt.xlabel("Number of Iterations", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.plot(losses)
plt.show()

# Compute predictions (Class 1 or Class 2)
scores = y_pred_grader(features, final_w, final_b)
pred_labels = (scores > 0.5).astype(int)
pred_labels[pred_labels != 1] = -1

plt.figure()

plt.title("Predicted Labels and Decision Boundary")
plt.xlabel("Feature 1", fontsize=12)
plt.ylabel("Feature 2", fontsize=12)
plt.axis([-3.3, 8.5, 1.6, 13.5])

# Plot the predictions (Class 1 or Class 2)
plt.scatter(
    features[pred_labels == 1, 0],  # x-coordinate
    features[pred_labels == 1, 1],  # y-coordinate
    label="Class 1 ($+1$)",
    color="blue",
    alpha=0.5,
)
plt.scatter(
    features[pred_labels != 1, 0],  # x-coordinate
    features[pred_labels != 1, 1],  # y-coordinate
    label="Class 2 ($-1$)",
    color="red",
    alpha=0.5,
)

# Plot the decision boundary
x = np.linspace(np.amin(features[:, 0]), np.amax(features[:, 0]), 10)
y = -(final_w[0] * x + final_b) / final_w[1]
plt.plot(x, y, "k--")

plt.legend(loc="upper left")
plt.show()
