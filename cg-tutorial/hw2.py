from __future__ import division

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math


def solve(A, b, x0=None, eps=1E-12, iter_max=1000, history=False):
    '''
    Solve system Ax=b by method suggest in Shewchuk HW2.

    If no initial guess x0 is provided zeros are used. The iteration stops if
    iteration cound exceeds iter_max or the current residum size is smaller
    than sqrt(eps)*r0, where r0 is the initial residuum.

    !!!NOTE THAT THE FORM MUST BY POSITIVE OR NEGATIVE DEFINITE!!!
    '''

    # Checks of input
    m, n = A.shape
    assert m == n
    assert b.shape == (n, )

    A = A.astype('float', copy=False)
    b = b.astype('float', copy=False)

    if x0 is None:
        x0 = np.zeros(n)

    n_iters = 0
    r = b - A.dot(x0)       # Initialize residuum
    r_norm = r.dot(r)
    r_norm0 = r_norm        # Remember size for stopping loop

    lambdas = iter(sorted(la.eigvals(A), reverse=True))

    if history:
        x0_history = [x0]

    while n_iters < iter_max and r_norm > eps*r_norm0:
        n_iters += 1

        # Step-size is the inverse of eigenvalue
        alpha = next(lambdas)**-1

        # Direction is the previous residual
        x0 = x0 + alpha*r
        if history:
            x0_history.append(x0)

        r = b - A.dot(x0)

        r_norm = r.dot(r)

    if history:
        return x0, n_iters, x0_history
    else:
        return x0, n_iters

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    A = np.array([[3, 2], [2, 6]])
    b = np.array([2, -8])
    x0 = np.array([-2, -2])

    x_np = la.solve(A, b)
    x, n_iters, x_path = solve(A, b, x0, eps=1E-4, history=True)

    print 'Numpy vs SG', la.norm(x_np - x)  # This should be a small number

    # Set up a grid around x_sd that contains x0 and plot the quadratic form
    w = math.ceil(max(map(abs, x - x0))) + 1

    X, Y = np.meshgrid(np.linspace(x[0]-w, x[0]+w, 100),
                       np.linspace(x[1]-w, x[1]+w, 100))

    # Generate values of qudratic function and its gradient
    PHI = np.zeros_like(X)
    phi = lambda x: 0.5*x.dot(A.dot(x)) - b.dot(x)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            P = np.array([X[i, j], Y[i, j]])
            PHI[i, j] = phi(P)

    # Let's get the form plot with contours
    fig = plt.figure()
    plt.pcolor(X, Y, PHI)
    plt.contour(X, Y, PHI, 10, colors='k')
    plt.axis('equal')

    # Plot all x from iteration
    for x in x_path:
        plt.plot(x[0], x[1], 'rx')

    # Plot the path
    for i in range(1, len(x_path)):
        P, Q = x_path[i-1], x_path[i]
        plt.plot(np.linspace(P[0], Q[0], 10),
                 np.linspace(P[1], Q[1], 10), 'r')

    plt.show()
