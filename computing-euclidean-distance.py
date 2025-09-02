import numpy as np
import time

import sys
sys.path.append("/home/codio/workspace/.modules")
from helper import *

print("Python version:", sys.version.split(" ")[0])

def L2distanceSlow(X, Z=None):
    """
    Computes the Euclidean distance (L2-distance) matrix very slowly
    using a naïve approach involving three nested for-loops.
    """
    if Z is None:
        Z = X

    n, d = X.shape        # Get shape of X
    m = Z.shape[0]        # Get number of rows in Z
    D = np.empty((n, m))  # Allocate memory for the output matrix
    
    # Use nested loops to compute the Euclidean distance matrix
    for i in range(n):          # Loop over vectors in X
        for j in range(m):      # Loop over vectors in Z
            for k in range(d):  # Loop over vector dimensions
                D[i, j] += (X[i, k] - Z[j, k]) ** 2
            
            # Compute Euclidean distance (L2-distance) between i-th and j-th vector    
            D[i, j] = np.sqrt(D[i, j])
    
    return D

X = np.random.rand(700, 100)
print("Running the slow naïve version, please wait...")
%time Dslow = L2distanceSlow(X)
print("This is NOT an efficient way to compute the distance matrix!")

def gram_matrix(X, Z=None):
    """
    Computes the Gram matrix G; i.e., computes a matrix of consisting of
    inner products between the row vectors of two input matrices X and Z.
    
    Syntax: G = gram_matrix(X, Z); you can also call this function
            with only one input, i.e., gram_matrix(X) is equivalent
            to gram_matrix(X, X).
    
    Input:
        X: nxd data matrix with n vectors (rows) of dimensionality d
        Z: mxd data matrix with m vectors (rows) of dimensionality d

    Output:
        G: nxm matrix for which each entry G[i,j] is the inner-product
           between vectors X[i,:] and Z[j,:]
    """
    if Z is None:
        Z = X

    G = np.dot(X, Z.T)

    return G

def gram_matrix_test1():
    """
    Check output shape of your gram_matrix function using one input matrix
    """
    X = np.random.rand(700, 10)  # Random input matrix X with 700 samples
    return gram_matrix(X).shape == (700, 700)


def gram_matrix_test2():
    """
    Check output shape of your gram_matrix function using two input matrices
    """
    X = np.random.rand(700, 10)  # Random input matrix X with 700 samples
    Z = np.random.rand(200, 10)  # Random input matrix Z with 200 samples
    return gram_matrix(X, Z).shape == (700, 200)


def gram_matrix_test3():
    """
    Check output accuracy of your gram_matrix function using one input matrix
    """
    np.random.seed(1)               # Set random seed for consistency
    X = np.random.rand(700, 100)    # Random input matrix X with 700 samples
    G1 = gram_matrix(X)             # Compute the Gram matrix with your code
    G2 = gram_matrix_grader(X)      # Compute the Gram matrix with our code
    diff = np.linalg.norm(G1 - G2)  # Compute the difference (using norm)
    return diff < 1e-5              # Difference should be small


def gram_matrix_test4():
    """
    Check output accuracy of your gram_matrix function using two input matrices
    """
    np.random.seed(2)               # Set random seed for consistency
    X = np.random.rand(700, 100)    # Random input matrix X with 700 samples
    Z = np.random.rand(300, 100)    # Random input matrix Z with 300 samples
    G1 = gram_matrix(X, Z)          # Compute the Gram matrix with your code
    G2 = gram_matrix_grader(X, Z)   # Compute the Gram matrix with our code
    diff = np.linalg.norm(G1 - G2)  # Compute the difference (using norm)
    return diff < 1e-5              # Difference should be small


runtest(gram_matrix_test1, "gram_matrix_test1")
runtest(gram_matrix_test2, "gram_matrix_test2")
runtest(gram_matrix_test3, "gram_matrix_test3")
runtest(gram_matrix_test4, "gram_matrix_test4")

def calculate_S(X, n, m):
    """
    function calculate_S(X)

    Computes the S matrix.
    Syntax:
    S=calculate_S(X)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    n: number of rows in X
    m: output number of columns in S

    Output:
    Matrix S of size nxm
    S[i,j] is the inner-product between vectors X[i,:] and X[i,:]
    """
    assert n == X.shape[0]

    S = np.einsum('ij,ij->i', X, X).reshape(n, 1) * np.ones((1, m))

    return S

def calculate_R(Z, n, m):
    """
    function calculate_R(Z)

    Computes the R matrix.
    Syntax:
    R=calculate_R(Z)
    Input:
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    n: output number of rows in Z
    m: number of rows in Z

    Output:
    Matrix R of size nxm
    R[i,j] is the inner-product between vectors Z[j,:] and Z[j,:]
    """
    assert m == Z.shape[0]
    
    R = np.einsum('ij,ij->i', Z, Z).reshape(1, m) * np.ones((n, 1))

    return R

def calculate_S_test1():
    """
    Check that the output of your calculate_S function has the correct shape
    """
    X = np.random.rand(700, 100)  # Random input matrix X
    Z = np.random.rand(800, 100)  # Random input matrix Z
    n, _ = X.shape                # Get number of rows in X
    m, _ = Z.shape                # Get number of rows in Z
    S = calculate_S(X, n, m)      # Compute your matrix S
    return S.shape == (n, m)      # Shape should be (n, m)


def calculate_S_test2():
    """
    Check that the output of your calculate_S function has the correct value
    """
    np.random.seed(1)                 # Set random seed for consistency
    X = np.random.rand(700, 100)      # Random input matrix X
    Z = np.random.rand(800, 100)      # Random input matrix Z
    n, _ = X.shape                    # Get number of rows in X
    m, _ = Z.shape                    # Get number of rows in Z
    S1 = calculate_S(X, n, m)         # Compute your matrix S
    S2 = calculate_S_grader(X, n, m)  # Compute our matrix S
    diff = np.linalg.norm(S1 - S2)    # Compute the difference
    return diff < 1e-5                # Difference should be small


def calculate_R_test1():
    """
    Check that the output of your calculate_R function has the correct shape
    """
    X = np.random.rand(700, 100)  # Random input matrix X
    Z = np.random.rand(800, 100)  # Random input matrix Z
    n, _ = X.shape                # Get number of rows in X
    m, _ = Z.shape                # Get number of rows in Z
    R = calculate_R(Z, n, m)      # Compute your matrix R
    return R.shape == (n, m)      # Shape should be (n, m) 


def calculate_R_test2():
    """
    Check that the output of your calculate_R function has the correct value
    """
    np.random.seed(2)                 # Set random seed for consistency
    X = np.random.rand(700, 100)      # Random input matrix X
    Z = np.random.rand(800, 100)      # Random input matrix Z
    n, _ = X.shape                    # Get number of rows in X
    m, _ = Z.shape                    # Get number of rows in Z
    R1 = calculate_R(Z, n, m)         # Compute your matrix R
    R2 = calculate_R_grader(Z, n, m)  # Compute our matrix R
    diff = np.linalg.norm(R1 - R2)    # Compute the difference
    return diff < 1e-5                # Difference should be small


runtest(calculate_S_test1, "calculate_S_test1")
runtest(calculate_S_test2, "calculate_S_test2")
runtest(calculate_R_test1, "calculate_R_test1")
runtest(calculate_R_test2, "calculate_R_test2")

def L2distance(X, Z=None):
    """
    Efficiently computes the Euclidean distance (L2-distance) matrix.
    
    Syntax: D = L2distance(X, Z); or you can call L2distance with only one
            input because D = L2distance(X) is equivalent to D = L2distance(X, X)
    
    Input:
        X: nxd data matrix with n vectors (rows) of dimensionality d
        Z: mxd data matrix with m vectors (rows) of dimensionality d

    Output:
        D: nxm matrix such that each entry D[i, j] is the Euclidean distance
           (L2-distance) between the two row vectors X[i, :] and Z[j, :]
    """
    if Z is None:
        Z = X

    n, d1 = X.shape
    m, d2 = Z.shape
    assert d1 == d2, "Dimensions of input vectors must match!"
    
    Xnorm = np.sum(X ** 2, axis=1).reshape(n, 1)
    Znorm = np.sum(Z ** 2, axis=1).reshape(1, m)
    D = np.sqrt(Xnorm + Znorm - 2 * np.dot(X, Z.T))

    return D

def L2distance_test1():
    """
    Check that your distance matrix has the correct shape
    """
    X = np.random.rand(700, 100)  # Random input matrix X
    Z = np.random.rand(800, 100)  # Random input matrix Z
    n, _ = X.shape                # Get number of rows in X
    m, _ = Z.shape                # Get number of rows in Z
    D = L2distance(X, Z)          # Compute your distance matrix
    return D.shape == (n, m)

def L2distance_test2():
    """
    Check your distance matrix against our distance matrix
    """
    np.random.seed(1)               # Set random seed for consistency
    X = np.random.rand(700, 100)    # Random input matrix X
    D1 = L2distance(X)              # Compute your distance matrix
    D2 = L2distance_grader(X)       # Compute our distance matrix
    diff = np.linalg.norm(D1 - D2)  # Compute the difference (using norm)
    return diff < 1e-5              # Difference should be small


def L2distance_test3():
    """
    Another check of your distance matrix against our distance matrix
    """
    np.random.seed(2)                 # Set random seed for consistency
    X = np.random.rand(700, 100)      # Random input matrix X
    D1 = L2distance(X)                # Compute your distance matrix
    D2sq = L2distance_grader(X) ** 2  # Compute our squared distance matrix
    diff = np.linalg.norm(D1 - D2sq)  # Compute the difference (using norm)
    return diff > 100                 # Difference should be big


def L2distance_test4():
    """
    Another check of your distance matrix against our distance matrix
    """
    np.random.seed(3)               # Set random seed for consistency
    X = np.random.rand(700, 100)    # Random input matrix X
    Z = np.random.rand(300, 100)    # Random input matrix Z
    D1 = L2distance(X, Z)           # Compute your distance matrix
    D2 = L2distance_grader(X, Z)    # Compute our distance matrix
    diff = np.linalg.norm(D1 - D2)  # Compute the difference (using norm)
    return diff < 1e-5              # Difference should be small


runtest(L2distance_test1, "L2distance_test1")
runtest(L2distance_test2, "L2distance_test2")
runtest(L2distance_test3, "L2distance_test3")
runtest(L2distance_test4, "L2distance_test4")

current_time = lambda: int(round(time.time() * 1000))

X = np.random.rand(700, 100)
Z = np.random.rand(300, 100)

print("Running the slow naïve version...")
before = current_time()
Dslow = L2distanceSlow(X)
after = current_time()
t_slow = after - before
print(f"{t_slow:.2f} ms\n")

print("Running the vectorized version...")
before = current_time()
Dfast = L2distance(X)
after = current_time()
t_fast = after - before
print(f"{t_fast:.2f} ms\n")

diff = np.linalg.norm(Dfast - Dslow)
speedup = t_slow / t_fast
print(f"The difference between the result of these two methods should be very small: {diff:.8f}")
print(f"However, your NumPy code was {speedup:.2f} times faster!")

