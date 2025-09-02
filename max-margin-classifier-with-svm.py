import numpy as np
from numpy.matlib import repmat
from scipy.stats import linregress
import time

# Interactive matplotlib plots
%matplotlib ipympl
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

xTr, yTr = generate_data()
visualize_2D(xTr, yTr)

def loss(w, b, xTr, yTr, C):
    """
    Calculates the squared hinge loss plus the L2-regularizer,
    as defined in the equation above.

    Input:
        w: d-dimensional weight vector
        b: Bias term (a scalar)
        xTr: nxd data matrix (each row is an input vector)
        yTr: n-dimensional vector (each entry is a label)
        C: Constant that controls the tradeoff between
           the L2-regularizer and hinge-loss (a scalar)

    Output:
        loss_val: Squared loss plus the L2-regularizer for
                  the classifier on xTr and yTr (a scalar)
    """
    loss_val = 0.0

    loss_val = np.sum(w.T*w) + C*np.sum(np.maximum((1 - yTr*(np.dot(w,xTr.T) + b)), 0)**2)

    return loss_val

# Generate data for testing your function loss() 
xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape


def loss_test1():
    """
    Check that function loss() returns a scalar
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)
    return np.isscalar(loss_val)


def loss_test2():
    """
    Check that loss() returns a non-negative scalar
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)
    return loss_val >= 0


def loss_test3():
    """
    Check that L2-regularizer is implemented correctly
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 0)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 0)
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


def loss_test4():
    """
    Check that squared hinge loss is implementedand and not standard hinge loss.
    Note: loss_grader_wrong is the wrong implementation of standard hinge loss,
    so the results should NOT match.
    """
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 1)
    badloss = loss_grader_wrong(w, b, xTr_test, yTr_test, 1)
    return not (np.linalg.norm(loss_val - badloss) < 1e-5)


def loss_test5():
    """
    Check that squared hinge loss is implemented correctly
    """
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 10)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 10)
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


def loss_test6():
    """
    Check that function loss() is implemented correctly
    """
    w = np.random.randn(d)
    b = np.random.rand(1)
    loss_val = loss(w, b, xTr_test, yTr_test, 100)
    loss_val_grader = loss_grader(w, b, xTr_test, yTr_test, 100)
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


runtest(loss_test1, "loss_test1")
runtest(loss_test2, "loss_test2")
runtest(loss_test3, "loss_test3")
runtest(loss_test4, "loss_test4")
runtest(loss_test5, "loss_test5")
runtest(loss_test6, "loss_test6")

def grad(w, b, xTr, yTr, C):
    """
    Calculates the gradients of the loss function,
    as given by the expressions above.

    Input:
        w: d-dimensional weight vector
        b: Bias term (a scalar)
        xTr: nxd data matrix (each row is an input vector)
        yTr: n-dimensional vector (each entry is a label)
        C: Constant that controls the tradeoff between
           the L2-regularizer and hinge loss (a scalar)
    """
    
    n, d = xTr.shape

    wgrad = np.zeros(d)
    bgrad = np.zeros(1)

    hinge_loss = np.maximum((1 - yTr*(np.dot(w,xTr.T) + b)), 0)
    wgrad = 2*w - 2*C * np.sum((hinge_loss * yTr).reshape(-1,1) * xTr, axis=0)
    bgrad = -2*C * np.sum(hinge_loss * yTr)

    return wgrad, bgrad

# Generate test data for testing grad()
xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape


def grad_test1():
    """
    Checks whether grad() returns a tuple of length 2
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    out = grad(w, b, xTr_test, yTr_test, 10)
    return type(out) == tuple and len(out) == 2


def grad_test2():
    """
    Checks the dimension of gradients
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    wgrad, bgrad = grad(w, b, xTr_test, yTr_test, 10)
    return len(wgrad) == d and np.isscalar(bgrad)


def grad_test3():
    """
    Checks the gradient of the L2-regularizer
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    wgrad, bgrad = grad(w, b, xTr_test, yTr_test, 0)
    wgrad_grader, bgrad_grader = grad_grader(w, b, xTr_test, yTr_test, 0)
    return (np.linalg.norm(wgrad - wgrad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


def grad_test4():
    """
    Checks the gradient of the squared hinge loss
    """
    w = np.zeros(d)
    b = np.random.rand(1)
    wgrad, bgrad = grad(w, b, xTr_test, yTr_test, 1)
    wgrad_grader, bgrad_grader = grad_grader(w, b, xTr_test, yTr_test, 1)
    return (np.linalg.norm(wgrad - wgrad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


def grad_test5():
    """
    Checks the gradient of the loss function
    """
    w = np.random.rand(d)
    b = np.random.rand(1)
    wgrad, bgrad = grad(w, b, xTr_test, yTr_test, 10)
    wgrad_grader, bgrad_grader = grad_grader(w, b, xTr_test, yTr_test, 10)
    return (np.linalg.norm(wgrad - wgrad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


runtest(grad_test1, "grad_test1")
runtest(grad_test2, "grad_test2")
runtest(grad_test3, "grad_test3")
runtest(grad_test4, "grad_test4")
runtest(grad_test5, "grad_test5")

w, b, final_loss = minimize(objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000)
print(f"The final loss of your model is {100 * final_loss:.2f}%.")
print(
    "The final squared hinge loss of your model is",
    f"{100 * (final_loss - np.dot(w, w)):.2f}%.",
)

# Calculate training error
err = np.mean(np.sign(xTr.dot(w) + b) != yTr)
print(f"Training Error: {100 * err:.2f}%")

# Visualize classifier and decision boundary
visualize_classfier(xTr, yTr, w, b)

# Initialize lists to store click data
Xdata = []
ldata = []

fig = plt.figure()
details = {
    "w": None,
    "b": None,
    "stepsize": 1,
    "ax": fig.add_subplot(111),
    "line": None,
}


def updateboundary(Xdata, ldata):
    global details
    w_pre, b_pre, _ = minimize(
        objective=loss,
        grad=grad,
        xTr=np.concatenate(Xdata),
        yTr=np.array(ldata),
        C=1000,
    )
    details["w"] = np.array(w_pre).reshape(-1)
    details["b"] = b_pre
    details["stepsize"] += 1


def updatescreen():
    global details
    b = details["b"]
    w = details["w"]
    q = -b / (w**2).sum() * w
    if details["line"] is None:
        (details["line"],) = details["ax"].plot(
            [q[0] - w[1], q[0] + w[1]], [q[1] + w[0], q[1] - w[0]], "k--"
        )
    else:
        details["line"].set_ydata([q[1] + w[0], q[1] - w[0]])
        details["line"].set_xdata([q[0] - w[1], q[0] + w[1]])


def generate_onclick(Xdata, ldata):
    global details

    def onclick(event):
        if event.key == "shift":  # Add negative point
            details["ax"].plot(event.xdata, event.ydata, "rx")
            label = -1
        else:  # Add positive point
            details["ax"].plot(event.xdata, event.ydata, "bo")
            label = 1
        pos = np.array([[event.xdata, event.ydata]])
        ldata.append(label)
        Xdata.append(pos)
        updateboundary(Xdata, ldata)
        updatescreen()

    return onclick


plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Add Points to Generate a Decision Boundary")
print("In the interactive plot below:")
print("1. Click to add a positive point")
print("2. Shift+click to add a negative point")
cid = fig.canvas.mpl_connect("button_press_event", generate_onclick(Xdata, ldata))
plt.show()

#scikit-learn implementation

from sklearn.svm import LinearSVC

clf = LinearSVC(
    penalty="l2",
    loss="squared_hinge",
    C=1000,
    max_iter=1000,
    random_state=0,
)
clf.fit(xTr, yTr)

# Calculate training error
err = np.mean(clf.predict(xTr) != yTr)
print(f"Training error: {100 * err:.2f}%")

# Visualize classifier and decision boundary
visualize_classfier(xTr, yTr, clf.coef_, clf.intercept_)
