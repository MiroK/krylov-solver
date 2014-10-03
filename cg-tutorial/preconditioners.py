from __future__ import division

import numpy as np
import numpy.linalg as la

class Preconditioner(object):
    'Preconditioner for symmetric posdef matrix.'
    def __init__(self, A):
        'Construct, get all the eigenvalues, and eigenvectors.'
        vals, vecs = la.eig(A)

        assert np.all(vals) > 0

        self.vals = 1./vals
        self.vecs = vecs.T

    def dot(self, x, B=None):
        '''
        Dot product of A^-1 and x. If B is given the action represents A^-1*B*x.
        '''
        y = np.zeros_like(x)
        if B is None:
            for val, vec in zip(self.vals, self.vecs):
                y += val*vec.dot(x)*vec
        else:
            for val, vec in zip(self.vals, self.vecs):
                y += val*(vec.dot(B).dot(x))*vec
        return y

if __name__ == '__main__':

    # Get symmetric pos def matrix
    n = 20
    while True:
        A = np.random.rand(n, n)
        A += A.T

        eigs = la.eigvals(A)
        if np.all(eigs) > 0:
            break

    # Get the inverse manually
    A_inv = la.inv(A)

    x = np.random.rand(n)
    # A^_1*x
    y = A_inv.dot(x)

    # Construct the A inverse
    B = Preconditioner(A)
    z = B.dot(x)

    # Difference should be small
    print la.norm(y - z)

    # A^-1 * C * z
    C = np.random.rand(n, n)
    yy = (A_inv.dot(C)).dot(z)
    # As action
    zz = B.dot(z, C)

    # Difference should be small
    print la.norm(yy - zz)
