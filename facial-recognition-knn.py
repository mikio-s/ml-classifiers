import numpy as np
from scipy.stats import mode
from scipy.io import loadmat
import time

# Interactive plots
%matplotlib ipympl
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

xTr, yTr, xTe, yTe = loaddata("faces.mat")

plt.figure(figsize=(9, 6))
plotfaces(xTr[:9, :])

def L2distance(X, Z=None):
    P = None
    if Z is None:
        Z = X
    n, d1 = X.shape
    m, d2 = Z.shape
    assert d1 == d2
    Xsqrd = np.sum(X ** 2, axis=1).reshape(n, 1)
    Zsqrd = np.sum(Z ** 2, axis=1).reshape(1, m)
    P = np.sqrt(Xsqrd + Zsqrd - 2 * np.dot(X, Z.T))
    return P

def findknn(xTr, xTe, k):
    """
    Finds the k-nearest neighbors of xTe in xTr.
    
    Syntax: (I, D) = findknn(xTr, xTe, k)

    Input:
        xTr: nxd input matrix with n row-vectors of dimensionality d
        xTe: mxd input matrix with m row-vectors of dimensionality d
        k: Number of nearest neighbors to be found

    Output:
        I: kxm matrix of indices, where I(i, j) is the i-th nearest neighbor of xTe(j, :)
        D: kxm matrix of Euclidean distances to the respective k-nearest neighbors
    """
    I = None
    D = None

    D = L2distance(xTr, xTe)
    I = np.argsort(D, axis=0)[:k, :]
    D = np.sort(D, axis=0)[:k, :]
    
    return I, D

def findknn_test1():
    """
    Check that your findknn function output has the correct types
    """
    xTr = np.random.rand(500, 10)  # 500 random training data points
    xTe = np.random.rand(300, 10)  # 300 random testing data points

    # Compute indices and distances to the 5-nearest neighbors
    Ig, Dg = findknn(xTr, xTe, 5)  
    
    # Check if Ig is a matrix of integers, Dg a matrix of floats
    result = (
        isinstance(Ig, np.ndarray)
        and isinstance(Ig, np.ndarray)
        and (isinstance(Dg[0][0], np.float64) or isinstance(Dg[0][0], np.float32))
        and (isinstance(Dg[0][0], np.float64) or isinstance(Dg[0][0], np.float32))
    )
    
    return result


def findknn_test2():
    """
    Check that your findknn function output has the correct shape
    """
    xTr = np.random.rand(500, 10)  # 500 random training data points
    xTe = np.random.rand(300, 10)  # 300 random testing data points
    
    # Compute indices and distances to the 5-nearest neighbors
    Ig, Dg = findknn(xTr, xTe, 5)

    # Check if output shape is correct
    result = Ig.shape == (5, 300) and Dg.shape == (5, 300)
    
    return result


def findknn_test3():
    """
    Check the accuracy of your findknn function for 1-NN
    """
    np.random.seed(1)              # Set random seed for consistency
    xTr = np.random.rand(500, 10)  # 500 random training data points
    xTe = np.random.rand(300, 10)  # 300 random testing data points
    
    # Compute indices and distances to the nearest neighbors with *your* code
    Ig, Dg = findknn(xTr, xTe, 1)  
    
    # Compute indices and distances to the nearest neighbors with *our* code
    Is, Ds = findknn_grader(xTr, xTe, 1)

    # Compute the difference between your output and our output
    result = np.linalg.norm(Ig - Is) + np.linalg.norm(Dg - Ds)
    
    return result < 1e-5


def findknn_test4():
    """
    Check the accuracy of your findknn function for 3-NN
    """
    np.random.seed(2)              # Set random seed for consistency
    xTr = np.random.rand(500, 10)  # 500 random training data points
    xTe = np.random.rand(300, 10)  # 300 random testing data points
    
    # Compute indices and distances to the nearest neighbors with *your* code
    Ig, Dg = findknn(xTr, xTe, 3)  
    
    # Compute indices and distances to the nearest neighbors with *our* code
    Is, Ds = findknn_grader(xTr, xTe, 3)

    # Compute the difference between your output and our output
    result = np.linalg.norm(Ig - Is) + np.linalg.norm(Dg - Ds)
    
    return result < 1e-5


runtest(findknn_test1, "findknn_test1")
runtest(findknn_test2, "findknn_test2")
runtest(findknn_test3, "findknn_test3")
runtest(findknn_test4, "findknn_test4")

k_demo1 = 5                        # Number of neighbors for this demo
N = 50                             # Number of training points
train_data = np.random.rand(N, 2)  # Random training dataset


def onclick(event):
    global k_demo1, train_data
    
    # Get coordinates of the test point
    if event.key == None:
        test_point = np.array([[event.xdata, event.ydata]])
        test_point_x = test_point[0, 0]  # x-coordinate of test point
        test_point_y = test_point[0, 1]  # y-coordinate of test point
        plt.scatter(test_point_x, test_point_y, color="red", marker="o")

    if event.key == "shift" and k_demo1 < N - 1:
        k_demo1 += 1

    if event.key == "control" and k_demo1 > 1:
        k_demo1 -= 1
        
    # Find k-nearest neighbors and get their indices
    indices, _ = findknn(train_data, test_point, k_demo1)

    # Add coordinates of the test point and k-nearest neighbors
    x_coords = []  # Initialize list of x-coordinates
    y_coords = []  # Initialize list of y-coordinates
    for i in range(k_demo1):
        x_coords.append(test_point_x)                  # x-coordinate of test point
        x_coords.append(train_data[indices[i, 0], 0])  # x-coordinate of neighboring point
        x_coords.append(None)                          # No x-coordinate (separator)
        y_coords.append(test_point_y)                  # y-coordinate of test point
        y_coords.append(train_data[indices[i, 0], 1])  # y-coordinate of neighboring point
        y_coords.append(None)                          # No y-coordinate (separator)

    # Plot the test point and k-nearest neighbors
    plt.plot(x_coords, y_coords, color="red", linestyle="-")
    if k_demo1 == 1:
        plt.title(f"{k_demo1}-Nearest Neighbor (Click to Add Another Test Point)")
    else:
        plt.title(f"{k_demo1}-Nearest Neighbors (Click to Add Another Test Point)")
    plt.show()


print("Please keep in mind:")
print("1. Run (or rerun) this cell to interact with the plot.")
print("2. Start the interactive demo by clicking the graph.")
print("3. Test points are not added to the training dataset.")
fig = plt.figure()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(train_data[:, 0], train_data[:, 1], color="b", marker="o")
plt.title(f"Click to Add a Test Point")
cid = fig.canvas.mpl_connect("button_press_event", onclick)

visualize_knn_images(findknn, imageType="faces")

def accuracy(truth, preds):
    """
    Analyzes the accuracy of a prediction against the ground truth

    Input:
        truth: n-dimensional vector of true class labels
        preds: n-dimensional vector of predictions

    Output:
        accur: Percent of predictions that are correct (a scalar)
    """
    accur = None
    
    t = np.array(truth).flatten()
    p = np.array(preds).flatten()
    accur = np.mean(t == p)
    
    return accur

def accuracy_test1():
    """
    Check that your accuracy function has the correct output type
    """
    truth = np.array([1, 2, 3, 4])
    preds = np.array([1, 2, 3, 0])
    result = isinstance(accuracy(truth, preds), np.float64)
    return result


def accuracy_test2():
    """
    Check your accuracy function on 4-sample data
    """
    truth = np.array([1, 2, 3, 4])
    preds = np.array([1, 2, 3, 0])
    return abs(accuracy(truth, preds) - 0.75) < 1e-10


def accuracy_test3():
    """
    Check your accuracy function on random data
    """
    np.random.seed(1)                         # Random seed for consistency
    p1 = np.random.rand(1, 1000)              # Random samples
    truth = np.int16(p1 > 0.5)                # Random string of 0s and 1s
    p2 = p1 + np.random.randn(1, 1000) * 0.1  # Add a little noise to p1
    preds = np.int16(p2 > 0.5)                # Very similar to truth
    diff = abs(accuracy(truth, preds) - accuracy_grader(truth, preds))
    return diff < 1e-10                       # Difference should be small


runtest(accuracy_test1, "accuracy_test1 (types)")
runtest(accuracy_test2, "accuracy_test2 (exactness)")
runtest(accuracy_test3, "accuracy_test3 (exactness)")


def knnclassifier(xTr, yTr, xTe, k):
    """
    Function that returns predictions from k-NN classifier

    Input:
        xTr: nxd input matrix with n row-vectors of dimensionality d
        yTr: n-dimensional vector of labels
        xTe: mxd input matrix with m row-vectors of dimensionality d
        k: Number of nearest neighbors to be found

    Output:
        preds: Predicted labels; i.e., preds[i] is the predicted label of xTe[i, :]
    """
    preds = None
    yTr = yTr.flatten()

    I, D = findknn(xTr, xTe, k)
    preds = mode(yTr[I], axis=0).mode.flatten()

    return preds

def knn_classifier_test1():
    X = np.array([[1, 0, 0, 1], [0, 1, 0, 1]]).T
    y = np.array([1, 1, 2, 2])
    preds = knnclassifier(X, y, X, 1)
    return isinstance(preds, np.ndarray) and preds.shape == (4,)


def knn_classifier_test2():
    X = np.array([[1, 0, 0, 1], [0, 1, 0, 1]]).T
    y = np.array([1, 1, 2, 2])
    return np.array_equal(knnclassifier(X, y, X, 1), y)


def knn_classifier_test3():
    X = np.array([[1, 0, 0, 1], [0, 1, 0, 1]]).T
    y = np.array([1, 1, 2, 2])
    y2 = np.array([2, 2, 1, 1])
    return np.array_equal(knnclassifier(X, y, X, 3), y2)


def knn_classifier_test4():
    X = np.array([[-4, -3, -2, 2, 3, 4]]).T
    y = np.array([1, 1, 1, 2, 2, 2])
    X2 = np.array([[-1, 1]]).T
    y2 = np.array([1, 2])
    return np.array_equal(knnclassifier(X, y, X2, 2), y2)


def knn_classifier_test5():
    X = np.array([[-4, -3, -2, 2, 3, 4]]).T
    y = np.array([1, 1, 1, 2, 2, 2])
    X2 = np.array([[0, 1]]).T
    y2 = np.array([1, 2])
    y3 = np.array([2, 2])
    result1 = np.array_equal(knnclassifier(X, y, X2, 2), y2)
    result2 = np.array_equal(knnclassifier(X, y, X2, 2), y3)
    return result1 or result2


def knn_classifier_test6():
    np.random.seed(1)
    X = np.random.rand(4, 4)
    y = np.array([1, 2, 2, 2])
    return accuracy(knnclassifier(X, y, X, 1), y) == 1


def knn_classifier_test7():
    np.random.seed(2)
    X = np.random.rand(4, 4)
    y = np.array([1, 2, 1, 2])
    return accuracy(knnclassifier(X, y, X, 1), y) == 1


def knn_classifier_test8():
    np.random.seed(3)
    X = np.random.rand(10, 100)
    y = np.round(np.random.rand(10)).astype("int")
    return accuracy(knnclassifier(X, y, X, 1), y) == 1


runtest(knn_classifier_test1, "knn_classifier_test1")
runtest(knn_classifier_test2, "knn_classifier_test2")
runtest(knn_classifier_test3, "knn_classifier_test3")
runtest(knn_classifier_test4, "knn_classifier_test4")
runtest(knn_classifier_test5, "knn_classifier_test5")
runtest(knn_classifier_test6, "knn_classifier_test6")
runtest(knn_classifier_test7, "knn_classifier_test7")
runtest(knn_classifier_test8, "knn_classifier_test8")

# Load the face data
xTr, yTr, xTe, yTe = loaddata("faces.mat")

# Make predictions and calculate accuracy
k = 1                                    # Use 1-NN classification
t0 = time.time()                         # Record start time
preds = knnclassifier(xTr, yTr, xTe, k)  # Make predictions
result = accuracy(yTe, preds)            # Compute accuracy
t1 = time.time()                         # Record stop time

print(f"Using {k}-NN classification to perform facial recognition...")
print(
    f"You obtained {100 * result:.2f}% classification",
    f"acccuracy in {t1 - t0:.4f} seconds.",
)

k_demo2 = 1
xy_data = np.empty((0, 2))
label_data = np.empty((0, 1))


def onclick_knn_classifier(event):
    global k_demo2, xy_data, label_data

    # Add positive point
    if event.key == None:
        label = 1
        point = np.array([event.xdata, event.ydata])
        xy_data = np.vstack((xy_data, point))
        label_data = np.append(label_data, label)

    # Add negative point
    if event.key == "n":
        label = -1
        point = np.array([event.xdata, event.ydata])
        xy_data = np.vstack((xy_data, point))
        label_data = np.append(label_data, label)
    
    # Increase `demo_k` by one
    if event.key == "shift" and k_demo2 < len(label_data) - 1:
        k_demo2 += 1
    
    # Decrease `demo_k` by one
    if event.key == "control" and k_demo2 > 1:
        k_demo2 -= 1

    RES = 50
    grid = np.linspace(0, 1, RES)
    X, Y = np.meshgrid(grid, grid)
    xTe = np.array([X.flatten(), Y.flatten()])
    Z = knnclassifier(xy_data, label_data, xTe.T, k_demo2)
    Z = Z.reshape(X.shape)
    Z[0, 0] = 1  # Optional: scale the colors correctly

    # Plot configuration
    plt.cla()                                  # Clear axes
    plt.xlim((0, 1))                           # x-axis limits
    plt.ylim((0, 1))                           # y-axis limits
    labels = ["$-1$", "$+1$"]                  # Labels for training points
    symbols = ["x", "o"]                       # "x" for -1 and "o" for +1 (training)
    mycolors = [[1, 0.5, 0.5], [0.5, 0.5, 1]]  # Red for -1 and blue for +1 (predictions)
    
    # Get the class values from training labels
    classvals = np.unique(label_data)

    # Plot prediction contours and data points
    if len(classvals) == 1 and 1 in classvals:
        # Plot blue prediction contours because only +1 points have been added
        plt.contourf(X, Y, np.sign(Z), colors=[[0.5, 0.5, 1], [0.5, 0.5, 1]])
        
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
        plt.contourf(X, Y, np.sign(Z), colors=mycolors)

        # Plot training data: "x" for -1 and "o" for +1
        for idx, c in enumerate(classvals):
            plt.scatter(
                xy_data[label_data == c, 0],  # x-coordinate of training point
                xy_data[label_data == c, 1],  # y-coordinate of training point
                label=labels[idx],            # Label of training point (-1 or +1)         
                marker=symbols[idx],          # Marker of training point ("x" or "o")
                color="black",                # Color of training point
            )
        
    plt.title(f"k-NN Classifier (k={k_demo2})")
    plt.legend(loc="upper left")
    plt.show()


# Interactive demo
print("Please keep in mind:")
print("1. Run (or rerun) this cell to interact with the plot.")
print("2. Start by clicking the graph to add a positive point.")
print("3. There is a delay between adding points or changing k.")
fig = plt.figure()
plt.title("Start by Clicking the Graph to Add a Positive Point")
plt.xlim(0, 1)
plt.ylim(0, 1)
cid = fig.canvas.mpl_connect("button_press_event", onclick_knn_classifier)

