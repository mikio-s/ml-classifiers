import numpy as np
import pandas as pd
import seaborn as sns

# Non-interactive plots
%matplotlib inline
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

def hashfeatures(baby, d, FIX, debug=False):
    """
    Input:
        baby: String representing the baby's name to be hashed
        d: Number of dimensions to be in the feature vector
        FIX: Number of chunks to extract and hash from each string
        debug: Boolean for printing debug values (default False)

    Output:
        v: Feature vector representing the input string
    """
    v = np.zeros(d)
    
    for m in range(1, FIX + 1):
        prefix = baby[:m] + ">"
        P = hash(prefix) % d
        v[P] = 1

        suffix = "<" + baby[-m:]
        S = hash(suffix) % d
        v[S] = 1

        if debug:
            print(f"Split {m}/{FIX}:\t({prefix}, {suffix}),\t1s at indices [{P}, {S}]")
    
    if debug:
        print(f"Feature vector for {baby}:\n{v.astype(int)}\n")

    return v

v = hashfeatures("Addisyn", d=128, FIX=3, debug=True)
v = hashfeatures("Addisyn", d=4, FIX=3, debug=True)
v = hashfeatures("Addisyn", d=128, FIX=7, debug=True)
v = hashfeatures("Max", d=128, FIX=4, debug=True)

def name2features(filename, d=128, FIX=3, LoadFile=True, debug=False):
    """
    Output:
        X : n feature vectors of dimension d, (nxd)
    """
    
    # Read in baby names
    if LoadFile:
        with open(filename) as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split("\n")
    
    n = len(babynames)
    X = np.zeros((n, d))
    
    for i in range(n):
        X[i, :] = hashfeatures(babynames[i], d, FIX)
        
    return (X, babynames) if debug else X

Xboys, namesBoys = name2features("boys.train", d=128, FIX=3, debug=True)
Xgirls, namesGirls = name2features("girls.train", d=128, FIX=3, debug=True)
X = np.concatenate([Xboys[:20], Xgirls[:20]], axis=0)

plt.figure(figsize=(12, 8))
ax = sns.heatmap(X.astype(int), cbar=False)
ax.set_xlabel("Feature Indices", fontsize=12)
ax.set_ylabel("Baby Names", fontsize=12)
ticks = ax.set_yticks(np.arange(40, dtype=int))
ticklabels = ax.set_yticklabels(namesBoys[:20] + namesGirls[:20])
plt.show()

def genTrainFeatures(dimension=128):
    """
    Input:
        dimension: Desired dimension of the features
    Output:
        X: n feature vectors of dimensionality d (nxd)
        Y: n labels (-1 = girl, +1 = boy) (n)
    """
    # Load in the data
    Xgirls = name2features("girls.train", d=dimension)
    Xboys = name2features("boys.train", d=dimension)
    X = np.concatenate([Xgirls, Xboys])

    # Generate labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])

    # Shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])

    return X[ii, :], Y[ii]

X, Y = genTrainFeatures(128)
print(f"Shape of training data: {X.shape}")
print(f"X:\n{X.astype(int)}")
print(f"Y:\n{Y.astype(int)}")

def naivebayesPY(X, Y):
    """
    naivebayesPY(X, Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """

    # Add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1, 1]])
    n = len(Y)
    # YOUR CODE HERE
    p_count = np.sum(Y == 1)
    n_count = np.sum(Y == -1)
    pos = p_count / n
    neg = n_count / n
    #print(pos)
    #print(neg)
    # END OF YOUR CODE

    return pos, neg

def naivebayesPY_test1():
    """
    Check that probabilities sum to 1
    """
    pos, neg = naivebayesPY(X, Y)
    return np.linalg.norm(pos + neg - 1) < 1e-5


def naivebayesPY_test2():
    """
    Test the naivebayesPY function on a simple example
    """
    x = np.array([[0, 1], [1, 0]])
    y = np.array([-1, 1])
    pos, neg = naivebayesPY(x, y)
    pos0, neg0 = 0.5, 0.5
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5


def naivebayesPY_test3():
    """
    Test the naivebayesPY function on another example
    """
    x = np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 1],
        ]
    )
    y = np.array([1, -1, 1, 1, -1, -1, 1])
    pos, neg = naivebayesPY(x, y)
    pos0, neg0 = 5 / 9.0, 4 / 9.0
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5


def naivebayesPY_test4():
    """
    Test plus-one smoothing
    """
    x = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]])
    y = np.array([1, 1])
    pos, neg = naivebayesPY(x, y)
    pos0, neg0 = 3 / 4.0, 1 / 4.0
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5


runtest(naivebayesPY_test1, "naivebayesPY_test1")
runtest(naivebayesPY_test2, "naivebayesPY_test2")
runtest(naivebayesPY_test3, "naivebayesPY_test3")
runtest(naivebayesPY_test4, "naivebayesPY_test4")

def naivebayesPXY(X, Y):
    """
    naivebayesPXY(X, Y) returns [posprob,negprob]

    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (n)

    Output:
        posprob: probability vector of p(x_alpha = 1|y=1)  (d)
        negprob: probability vector of p(x_alpha = 1|y=-1) (d)
    """

    # Add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = X.shape
    X = np.concatenate([X, np.ones((2, d)), np.zeros((2, d))])
    Y = np.concatenate([Y, [-1, 1, -1, 1]])

    p = (Y == 1)
    n = (Y == -1)
    X_p = X[p]
    X_n = X[n]
    posprob = np.sum(X_p, axis=0) / X_p.shape[0]
    negprob = np.sum(X_n, axis=0) / X_n.shape[0]
    print(posprob)
    print(negprob)
    
  return posprob, negprob

X, Y = genTrainFeatures(128)
posprob, negprob = naivebayesPXY(X, Y)
probs = pd.DataFrame(
    {"feature": np.arange(128, dtype=int), "boys": posprob, "girls": negprob}
)

plt.figure(figsize=(12, 4))
ax = sns.lineplot(
    x="feature", y="value", hue="variable", data=pd.melt(probs, ["feature"])
)
ax.set_xlabel("Feature Indices", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)
plt.show()

def naivebayesPXY_test1():
    """
    Test a simple toy example with two points (one positive, one negative)
    """
    x = np.array([[0, 1], [1, 0]])
    y = np.array([-1, 1])
    pos, neg = naivebayesPXY(x, y)
    pos0, neg0 = naivebayesPXY_grader(x, y)
    test = np.linalg.norm(pos - pos0) + np.linalg.norm(neg - neg0)
    return test < 1e-5


def naivebayesPXY_test2():
    """
    Test the probabilities P(X|Y=+1)
    """
    pos, neg = naivebayesPXY(X, Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(pos - posprobXY)
    return test < 1e-5


def naivebayesPXY_test3():
    """
    Test the probabilities P(X|Y=-1)
    """
    pos, neg = naivebayesPXY(X, Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    test = np.linalg.norm(neg - negprobXY)
    return test < 1e-5


def naivebayesPXY_test4():
    """
    Check that the dimensions of the posterior probabilities are correct
    """
    pos, neg = naivebayesPXY(X, Y)
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    return pos.shape == posprobXY.shape and neg.shape == negprobXY.shape


runtest(naivebayesPXY_test1, "naivebayesPXY_test1")
runtest(naivebayesPXY_test2, "naivebayesPXY_test2")
runtest(naivebayesPXY_test3, "naivebayesPXY_test3")
runtest(naivebayesPXY_test4, "naivebayesPXY_test4")

def loglikelihood(posprob, negprob, X_test, Y_test):
    """
    Function: loglikelihood(posprob, negprob, X_test, Y_test)
    Returns: Log-likelihood of each point in X_test

    Input:
        posprob: Conditional probabilities for the positive class (d)
        negprob: Conditional probabilities for the negative class (d)
        X_test: Features (nxd)
        Y_test: Labels (-1 or +1) (n)

    Output:
        loglikelihood: Log-likelihood of each point in X_test (n)
    """
    n, d = X_test.shape
    loglikelihood = np.zeros(n)
    
    loglikelihood[Y_test > 0] = np.dot(X_test[Y_test > 0], np.log(posprob)) + np.dot((1 - X_test[Y_test > 0]), np.log(1 - posprob))
    loglikelihood[Y_test < 0] = np.dot(X_test[Y_test < 0], np.log(negprob)) + np.dot((1 - X_test[Y_test < 0]), np.log(1 - negprob))
    print(loglikelihood)

    return loglikelihood

X, Y = genTrainFeatures(128)
posprob, negprob = naivebayesPXY_grader(X, Y)


def loglikelihood_test1():
    """
    Test if the log-likelihood of the training data are all negative
    """
    ll = loglikelihood(posprob, negprob, X, Y)
    return all(ll < 0)


def loglikelihood_test2():
    """
    Test if the log-likelihood of the training data matches the solution
    """
    ll = loglikelihood(posprob, negprob, X, Y)
    llgrader = loglikelihood_grader(posprob, negprob, X, Y)
    return np.linalg.norm(ll - llgrader) < 1e-5


def loglikelihood_test3():
    """
    Test if the log-likelihood of the training data matches the solution
    (positive points only)
    """
    ll = loglikelihood(posprob, negprob, X, Y)
    llgrader = loglikelihood_grader(posprob, negprob, X, Y)
    return np.linalg.norm(ll[Y == 1] - llgrader[Y == 1]) < 1e-5


def loglikelihood_test4():
    """
    Test if the log-likelihood of the training data matches the solution
    (negative points only)
    """
    ll = loglikelihood(posprob, negprob, X, Y)
    llgrader = loglikelihood_grader(posprob, negprob, X, Y)
    return np.linalg.norm(ll[Y == -1] - llgrader[Y == -1]) < 1e-5


def loglikelihood_test5():
    """
    Little toy example with two data points (1 positive, 1 negative)
    """
    x = np.array([[0, 1], [1, 0]])
    y = np.array([-1, 1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:2], negprobXY[:2], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:2], negprobXY[:2], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


def loglikelihood_test6():
    """
    Little toy example with four data points (2 positive, 2 negative)
    """
    x = np.array(
        [[1, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1]]
    )
    y = np.array([-1, 1, 1, -1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


def loglikelihood_test7():
    """
    One more toy example with 5 positive and 2 negative points
    """
    x = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1],
        ]
    )
    y = np.array([1, 1, 1, 1, -1, -1, 1])
    posprobXY, negprobXY = naivebayesPXY_grader(X, Y)
    loglike = loglikelihood(posprobXY[:6], negprobXY[:6], x, y)
    loglike0 = loglikelihood_grader(posprobXY[:6], negprobXY[:6], x, y)
    test = np.linalg.norm(loglike - loglike0)
    return test < 1e-5


runtest(
    loglikelihood_test1,
    "loglikelihood_test1 (all log-likelihoods must be negative)",
)
runtest(loglikelihood_test2, "loglikelihood_test2 (training data)")
runtest(loglikelihood_test3, "loglikelihood_test3 (positive points)")
runtest(loglikelihood_test4, "loglikelihood_test4 (negative points)")
runtest(loglikelihood_test5, "loglikelihood_test5")
runtest(loglikelihood_test6, "loglikelihood_test6")
runtest(loglikelihood_test7, "loglikelihood_test7")

def naivebayes_pred(pos, neg, posprob, negprob, X_test):
    """
    Function: naivebayes_pred(pos, neg, posprob, negprob, X_test)
    Returns: The prediction of each point in X_test

    Input:
        pos: Class probability for the negative class
        neg: Class probability for the positive class
        posprob: Conditional probabilities for the positive class (d)
        negprob: Conditional probabilities for the negative class (d)
        X_test: Features (nxd)

    Output:
        preds: Prediction of each point in X_test (n)
    """
    n, d = X_test.shape
    preds = np.empty(n)
    
    Y_p = np.ones(n)
    Y_n = Y_p * -1
    positive = loglikelihood(posprob, negprob, X_test, Y_p) + np.log(pos) - np.log(neg)
    negative = loglikelihood(posprob, negprob, X_test, Y_n) + np.log(neg) - np.log(pos)
    dif = positive - negative
    preds[dif > 0] = 1
    preds[dif < 0] = -1

    return preds

X, Y = genTrainFeatures_grader(128)
posY, negY = naivebayesPY_grader(X, Y)
posprobXY, negprobXY = naivebayesPXY_grader(X, Y)


def naivebayes_pred_test1():
    """
    Check whether the predictions are +1 or neg 1
    """
    preds = naivebayes_pred(posY, negY, posprobXY, negprobXY, X)
    return np.all(np.logical_or(preds == -1, preds == 1))


def naivebayes_pred_test2():
    x_test = np.array([[0, 1], [1, 0]])
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:2], negprobXY[:2], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5


def naivebayes_pred_test3():
    x_test = np.array(
        [
            [1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ]
    )
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5


def naivebayes_pred_test4():
    x_test = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1],
        ]
    )
    preds = naivebayes_pred_grader(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    student_preds = naivebayes_pred(posY, negY, posprobXY[:6], negprobXY[:6], x_test)
    acc = analyze_grader("acc", preds, student_preds)
    return np.abs(acc - 1) < 1e-5


runtest(naivebayes_pred_test1, "naivebayes_pred_test1")
runtest(naivebayes_pred_test2, "naivebayes_pred_test2")
runtest(naivebayes_pred_test3, "naivebayes_pred_test3")
runtest(naivebayes_pred_test4, "naivebayes_pred_test4")

DIMS = 128
print("Loading data...")
X, Y = genTrainFeatures(DIMS)
print("Training classifier...")
pos, neg = naivebayesPY(X, Y)
posprob, negprob = naivebayesPXY(X, Y)
error = np.mean(naivebayes_pred(pos, neg, posprob, negprob, X) != Y)
print(f"Training error: {100 * error:.2f}%")

while True:
    print("Please enter a baby name (press Enter key with empty box to stop prompt)>")
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname, d=DIMS, LoadFile=False)
    pred = naivebayes_pred(pos, neg, posprob, negprob, xtest)
    if pred > 0:
        print(f"{yourname}, I am sure you are a baby boy.\n")
    else:
        print(f"{yourname}, I am sure you are a baby girl.\n")

  
