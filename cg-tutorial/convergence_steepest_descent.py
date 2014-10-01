from __future__ import division
import numpy as np
import numpy.linalg as la
from steepest_descent import solve

'''
We shall consider here a system Ax=b with A, a 2x2 symmetric positive
definite matrix. The system is solved by the steepest descent method
where we have that
    
    (e_{i+1}, e_{i+1})_A = w*2(e_i, e_i)_A
    
and w*2 is given as

    w*2 = (mu**2 + k**2)**2/(mu**2 + k**3)/(mu**2 + k)

where k is the spectral condition number, i.e k=lambda_max/lamnda_min
and mu is (u_min, e_i)/(u_max, e_i).

1)
Thus, is the initial error is the eigenvector of A, mu is 0 or infty and
convergence is immediate. VERIFY

2)
For general initial guess, the convergence get worse of large condition
number. VERIFY

'''

def sd_convergence(A, b, x0, x):
    '''
    Study convergence of steepest descent method based on x0 and condition
    number of A.
    '''
   
    assert A.shape == (2, 2), 'A is not a 2x2 matrix'
    assert b.shape == (2, ), 'b is not a vector of len 2'
    assert x0.shape == (2, ), 'x0 is not a vector of len 2'
    assert x.shape == (2, ), 'x0 is not a vector of len 2'

    assert abs(A[0, 1] - A[1, 0]) < 1E-15, 'A is not positive definite'

    lambdas, vecs = la.eigs(A)
    assert np.all(lmabdas > 0), 'A is not positive definite'

    lambda0, lambda1 = lambdas
    u0, u1 = vecs[:, 0], vecs[:, 1]

    # Swap if not sorted
    if lambda0 > lambda1:
        lambda1, lambd0 = lambda0, lambda1
        u1, u0 = u0, u1

    # error
    e0 = x0 - x

    k = lambda1/lambda0
    mu = u0.dot(e0)/u1.dot(e0)

    print 'Condition number is', k
    print 'Initial guess yields mu', mu

    X, n_iters = solve(A, b, x0)

    print 'error in solution', la.norm(X - x)
    print n_iters

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    b = np.zeros(2)
    x = np.zeros(2)

    # Small condition number
    A_small = np.array([[2, 0], [0, 1]]) 

    # Large condition number
    A_large = np.array([[2, 0], [0, 1E-6]])
   
    # This gives error in dir of eig
    x_eig = np.array([1, 0])

    # This does not
    x_gen = np.array([1, 1])

    print solve(A_small, b, x_eig)  # Done in 1 iter
    print solve(A_small, b, x_gen)  # Done in 11 iter
    print solve(A_large, b, x_eig)  # Done in 1 iter
    print solve(A_large, b, x_gen)  # Converges to wrong result!

