import numpy as np

# Interactive matplotlib plots
%matplotlib ipympl
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

xTr, yTr = generate_data()
visualize_2D(xTr, yTr)

def computeK(kerneltype, X, Z, kpar=1):
    """
    Computes a matrix K such that K[i, j] = K(x_i, z_j).
    The kernel operation is defined by kerneltype with parameter kpar.

    Input:
        kerneltype: "linear", "polynomial", or "rbf'"
        X: nxd data matrix
        Z: mxd data matrix
        kpar: Kernel parameter (inverse sigma^2 for "rbf", degree p for "polynomial")

    Output:
        K: nxm kernel matrix
    """
    assert kerneltype in ["linear", "polynomial", "rbf"], (
        "Kernel type %s not known." % kerneltype
    )
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"

    K = None

    if kerneltype == "linear":
        K=X @ Z.T
    if kerneltype == "rbf":
        K=np.exp(-kpar * l2distance(X,Z)**2)
        
    if kerneltype == "polynomial":
        K=np.power(1+X @ Z.T, kpar)

    return K

# Generate data for testing kernel matrices
xTr_test, yTr_test = generate_data(100)
xTr_test2, yTr_test2 = generate_data(50)
n, d = xTr_test.shape


def computeK_test1():
    """
    Checks that kernel matrices have the correct shape
    """
    s1 = computeK("rbf", xTr_test, xTr_test2, kpar=1).shape == (100, 50)
    s2 = computeK("polynomial", xTr_test, xTr_test2, kpar=1).shape == (100, 50)
    s3 = computeK("linear", xTr_test, xTr_test2, kpar=1).shape == (100, 50)
    return s1 and s2 and s3


def computeK_test2():
    """
    Checks that kernel matrices are symmetric
    """
    k_rbf = computeK("rbf", xTr_test, xTr_test, kpar=1)
    s1 = np.allclose(k_rbf, k_rbf.T)

    k_poly = computeK("polynomial", xTr_test, xTr_test, kpar=1)
    s2 = np.allclose(k_poly, k_poly.T)

    k_linear = computeK("linear", xTr_test, xTr_test, kpar=1)
    s3 = np.allclose(k_linear, k_linear.T)

    return s1 and s2 and s3


def computeK_test3():
    """
    Checks that kernel matrices are positive semi-definite
    """
    k_rbf = computeK("rbf", xTr_test2, xTr_test2, kpar=1)
    eigen_rbf = np.linalg.eigvals(k_rbf)
    eigen_rbf[np.isclose(eigen_rbf, 0)] = 0
    s1 = np.all(eigen_rbf >= 0)

    k_poly = computeK("polynomial", xTr_test2, xTr_test2, kpar=1)
    eigen_poly = np.linalg.eigvals(k_poly)
    eigen_poly[np.isclose(eigen_poly, 0)] = 0
    s2 = np.all(eigen_poly >= 0)

    k_linear = computeK("linear", xTr_test2, xTr_test2, kpar=1)
    eigen_linear = np.linalg.eigvals(k_linear)
    eigen_linear[np.isclose(eigen_linear, 0)] = 0
    s3 = np.all(eigen_linear >= 0)

    return s1 and s2 and s3


def computeK_test4():
    """
    Checks that rbf kernel matrix is correct
    """
    k = computeK("rbf", xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader("rbf", xTr_test, xTr_test2, kpar=1)
    return np.linalg.norm(k - k2) < 1e-5


def computeK_test5():
    """
    Checks that polynomial kernel matrix is correct
    """
    k = computeK("polynomial", xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader("polynomial", xTr_test, xTr_test2, kpar=1)
    return np.linalg.norm(k - k2) < 1e-5


def computeK_test6():
    """
    Checks that linear kernel matrix is correct
    """
    k = computeK("linear", xTr_test, xTr_test2, kpar=1)
    k2 = computeK_grader("linear", xTr_test, xTr_test2, kpar=1)
    return np.linalg.norm(k - k2) < 1e-5


runtest(computeK_test1, "computeK_test1")
runtest(computeK_test2, "computeK_test2")
runtest(computeK_test3, "computeK_test3")
runtest(computeK_test4, "computeK_test4")
runtest(computeK_test5, "computeK_test5")
runtest(computeK_test6, "computeK_test6")

def loss(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    """
    Calculates the loss (regularizer + squared hinge loss) for
    testing data against training data and parameters beta, b.

    Input:
        beta: n-dimensional vector that stores the linear combination coefficients
        b: Bias term (a scalar)
        xTr: nxd dimensional data matrix (training set, each row is an input vector)
        yTr: n-dimensional vector (training labels, each entry is a label)
        xTe: mxd dimensional matrix (test set, each row is an input vector)
        yTe: m-dimensional vector (test labels, each entry is a label)
        C: Scalar (constant that controls the tradeoff between L2-regularizer and hinge loss)
        kerneltype: Can be "linear", "polynomial", or "rbf"
        kpar: Kernel parameter (inverse sigma^2 for "rbf", degree p for "polynomial")

    Output:
        loss_val: Total loss (a scalar) obtained with (beta, xTr, yTr, b) on xTe and yTe
    """
    loss_val = 0.0

    # Compute the kernel values between xTr and xTr
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)

    # Compute the kernel values between xTr and xTe
    kernel_test = computeK(kerneltype, xTr, xTe, kpar)

    reg = np.dot(np.dot(kernel_train, beta), beta)
    pred = np.dot(kernel_test, beta) + b
    hinge_loss = np.sum(np.maximum((1 - yTr*pred), 0)**2)
    loss_val = reg+C*hinge_loss

    return loss_val

# Generate data for testing your function loss()
xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape


def loss_test1():
    """
    Check that loss() returns a scalar
    """
    beta = np.zeros(n)
    b = np.zeros(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, "rbf")
    return np.isscalar(loss_val)


def loss_test2():
    """
    Check that loss() returns a non-negative scalar
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, "rbf")
    return loss_val >= 0


def loss_test3():
    """
    Check that the L2-regularizer is implemented correctly
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 0, "rbf")
    loss_val_grader = loss_grader(
        beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 0, "rbf"
    )
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


def loss_test4():
    """
    Check that the square hinge loss is implemented correctly
    """
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, "rbf")
    loss_val_grader = loss_grader(
        beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 10, "rbf"
    )
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


def loss_test5():
    """
    Check that loss() is implemented correctly
    """
    beta = np.zeros(n)
    b = np.random.rand(1)
    loss_val = loss(beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 100, "rbf")
    loss_val_grader = loss_grader(
        beta, b, xTr_test, yTr_test, xTr_test, yTr_test, 100, "rbf"
    )
    return np.linalg.norm(loss_val - loss_val_grader) < 1e-5


runtest(loss_test1, "loss_test1")
runtest(loss_test2, "loss_test2")
runtest(loss_test3, "loss_test3")
runtest(loss_test4, "loss_test4")
runtest(loss_test5, "loss_test5")

def grad(beta, b, xTr, yTr, C, kerneltype, kpar=1):
    """
    Calculates the gradients of the loss function with respect to beta and b.

    Input:
        beta: n-dimensional vector that stores the linear combination coefficients
        b: Bias term (a scalar)
        xTr: nxd dimensional data matrix (training set, each row is an input vector)
        yTr: n-dimensional vector (training labels, each entry is a label)
        C: Scalar that controls the tradeoff between L2-regularizer and hinge loss
        kerneltype: Either "linear", "polynomial", or "rbf"
        kpar: Kernel parameter (inverse of sigma^2 for "rbf", degree p for "polynomial")

    Output:
        beta_grad, bgrad
        beta_grad:  n-dimensional vector (the gradient of loss with respect to the vector beta)
        bgrad:  Scalar (the gradient of loss with respect to the bias term b)
    """
    n, d = xTr.shape

    beta_grad = np.zeros(n)
    bgrad = np.zeros(1)

    # Compute the kernel values between xTr and xTr
    kernel_train = computeK(kerneltype, xTr, xTr, kpar)

    pred = np.dot(kernel_train, beta) + b
    hinge_loss = np.maximum((1 - yTr*pred), 0)
    beta_grad = 2*np.dot(kernel_train, beta) - 2*C * np.sum((hinge_loss * yTr).reshape(-1,1) * kernel_train, axis=0)
    bgrad = -2*C * np.sum(hinge_loss * yTr)

    return beta_grad, bgrad

# Generate data for testing your function grad()
xTr_test, yTr_test = generate_data()
n, d = xTr_test.shape


def grad_test1():
    """
    Checks that grad() returns a tuple of length 2
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    out = grad(beta, b, xTr_test, yTr_test, 10, "rbf")
    return type(out) == tuple and len(out) == 2


def grad_test2():
    """
    Checks the dimension of gradients
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 10, "rbf")
    return len(beta_grad) == n and np.isscalar(bgrad)


def grad_test3():
    """
    Checks the gradient of the L2-regularizer
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 0, "rbf")
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 0, "rbf")
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


def grad_test4():
    """
    Checks the gradient of the squared hinge loss
    """
    beta = np.zeros(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 1, "rbf")
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 1, "rbf")
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


def grad_test5():
    """
    Checks the gradient of the loss
    """
    beta = np.random.rand(n)
    b = np.random.rand(1)
    beta_grad, bgrad = grad(beta, b, xTr_test, yTr_test, 10, "rbf")
    beta_grad_grader, bgrad_grader = grad_grader(beta, b, xTr_test, yTr_test, 10, "rbf")
    return (np.linalg.norm(beta_grad - beta_grad_grader) < 1e-5) and (
        np.linalg.norm(bgrad - bgrad_grader) < 1e-5
    )


runtest(grad_test1, "grad_test1")
runtest(grad_test2, "grad_test2")
runtest(grad_test3, "grad_test3")
runtest(grad_test4, "grad_test4")
runtest(grad_test5, "grad_test5")

beta_sol, bias_sol, final_loss = minimize(
    objective=loss, grad=grad, xTr=xTr, yTr=yTr, C=1000, kerneltype="linear", kpar=1
)
print(f"The final loss of your model is {final_loss:.4f}.")

K_nn = computeK("linear", xTr, xTr, kpar=1)
reg = beta_sol @ K_nn @ beta_sol
print(f"The final squared hinge loss of your model is {final_loss - reg:.4f}.")

svmclassify = lambda x: np.sign(computeK("linear", x, xTr, 1).dot(beta_sol) + bias_sol)
predsTr = svmclassify(xTr)
training_err = np.mean(np.sign(predsTr) != yTr)
print(f"Training Error: {100 * training_err:.2f}%")

visclassifier(svmclassify, xTr, yTr)

xTr_spiral, yTr_spiral, xTe_spiral, yTe_spiral = spiraldata()
visualize_2D(xTr_spiral, yTr_spiral)

beta_sol_spiral, bias_sol_spiral, final_loss_spiral = minimize(
    objective=loss,
    grad=grad,
    xTr=xTr_spiral,
    yTr=yTr_spiral,
    C=100,
    kerneltype="rbf",
    kpar=1,
)
print(f"The final loss of your model is {final_loss_spiral:.4f}.")

K_nn = computeK("rbf", xTr_spiral, xTr_spiral, kpar=1)
reg = beta_sol_spiral @ K_nn @ beta_sol_spiral
print(f"The final squared hinge loss of your model is {final_loss_spiral - reg:.4f}.")

svmclassify_spiral = lambda x: np.sign(
    computeK("rbf", xTr_spiral, x, 1).transpose().dot(beta_sol_spiral) + bias_sol_spiral
)

predsTr_spiral = svmclassify_spiral(xTr_spiral)
training_err_spiral = np.mean(predsTr_spiral != yTr_spiral)
print(f"Training Error: {100 * training_err_spiral:.2f}%")

predsTe_spiral = svmclassify_spiral(xTe_spiral)
test_err_spiral = np.mean(predsTe_spiral != yTe_spiral)
print(f"Test Error: {100 * test_err_spiral:.2f}%")

visclassifier(svmclassify_spiral, xTr_spiral, yTr_spiral)

# Initialize empty array of width 2 to store training points
xy_data = np.empty((0, 2))

# Initialize empty array of width 1 to store training labels
label_data = np.empty((0, 1))


def onclick_classifier(event):
    """
    Visualize kernel SVM classifier by adding new points
    """
    global xy_data, label_data

    # Click to add a positive point
    if event.key is None:
        label = 1
    
    # Shift+click to add a negative point
    if event.key == "shift":
        label = -1

    # Create position vector for new point
    pos = np.array([[event.xdata, event.ydata]])

    # Add new point and label to training data
    xy_data = np.vstack((xy_data, pos))
    label_data = np.append(label_data, label)

    # Create kernel SVM function
    svmC = 10
    beta_sol, bias_sol, final_loss = minimize(
        objective=loss,
        grad=grad,
        xTr=xy_data,
        yTr=label_data,
        C=svmC,
        kerneltype="rbf",
        kpar=1,
    )
    func = lambda x: np.sign(
        computeK("rbf", xy_data, x, 1).transpose().dot(beta_sol) + bias_sol
    )

    # Return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(0, 1, res)
    yrange = np.linspace(0, 1, res)

    # Repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    # Test all points on the grid
    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T
    testpreds = func(xTe)

    # Reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    Z[0, 0] = 1  # Optional: scale the colors correctly

    # Plot configuration
    plt.cla()                                  # Clear axes
    plt.xlim((0, 1))                           # x-axis limits
    plt.ylim((0, 1))                           # y-axis limits
    labels = ["$-1$", "$+1$"]                  # Labels for training points
    symbols = ["x", "o"]                       # "x" for -1 and "o" for +1 (training)
    mycolors = [[1, 0.5, 0.5], [0.5, 0.5, 1]]  # Red for -1 and blue for +1 (predictions)

    # Get class values from training labels
    classvals = np.unique(label_data)
    
    if len(classvals) == 1 and 1 in classvals:
        # Plot blue prediction contours because only +1 points have been added
        plt.contourf(pixelX, pixelY, np.sign(Z), colors=[[0.5, 0.5, 1], [0.5, 0.5, 1]])
        
        # Plot points with "o" markers because only +1 points have been added
        for idx, c in enumerate(classvals):
            plt.scatter(
                xy_data[label_data == c, 0],  # x-coordinate of training point
                xy_data[label_data == c, 1],  # y-coordinate of training point
                label="$+1$",                 # Label of training point
                marker="o",                   # Marker of training point
                color="black",                # Color of training point
            )
    else:
        # Plot prediction contours: red for -1 and blue for +1
        plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

        # Plot training data: "x" for -1 and "o" for +1
        for idx, c in enumerate(classvals):
            plt.scatter(
                xy_data[label_data == c, 0],  # x-coordinate of training point
                xy_data[label_data == c, 1],  # y-coordinate of training point
                label=labels[idx],            # Label of training point (-1 or +1)         
                marker=symbols[idx],          # Marker of training point ("x" or "o")
                color="black",                # Color of training point
            )

    plt.title("Click: Positive Point, Shift+Click: Negative Point")
    plt.legend(loc="upper left")
    plt.show()


# Interactive demo
print("Please keep in mind:")
print("1. You must run (or rerun) this cell right before interacting with the plot.")
print("2. Start the interactive demo by clicking the grid to add a positive point.")
print("3. Click to add a positive point or shift+click to add a negative point.")
print("4. You may notice a slight delay when adding points to the visualization.")
fig = plt.figure()
plt.title("Start by Clicking the Grid to Add a Positive Point")
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect("button_press_event", onclick_classifier)

#Scikit-Learn Implementation

from sklearn.svm import SVC

# Define model
clf = SVC(
    C=100,
    kernel="rbf",
    gamma=1,
    shrinking=False,
    tol=1e-8,
    max_iter=10000,
    random_state=0,
)

# Fit model on spiral dataset
clf.fit(xTr_spiral, yTr_spiral)

# Training error
predsTr_spiral = clf.predict(xTr_spiral)
trainingerr_spiral = np.mean(predsTr_spiral != yTr_spiral)
print(f"Training Error: {100 * trainingerr_spiral:.2f}%")

# Test error
predsTe_spiral = clf.predict(xTe_spiral)
testerr_spiral = np.mean(predsTe_spiral != yTe_spiral)
print(f"Test Error: {100 * testerr_spiral:.2f}%")

