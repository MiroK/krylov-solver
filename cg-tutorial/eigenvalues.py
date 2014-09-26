import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

A = np.array([[1, 1], [1, 2]])
vals, vecs = la.eig(A)

print 'Eigenvalues', vals
print 'Eigenvectors', vecs[:, 0], vecs[:, 1]

vecs = vecs.T

# We want to show that A^n*u where u is eigenvector of |lambda| < 1 will go
# to zero. If |lambda| > 1 will go to infinity
for val, vec in zip(vals, vecs):
    print 'Eigenvalue', val
    for n in range(10):
        vec = A.dot(vec)
        print '\t', la.norm(vec)

# We want to show that A^n*x, x is some random vector, is determined by
# the eigenvectors
x = np.array([2, 2])

x_e0 = x.dot(vecs[0])
x_e1 = x.dot(vecs[1])

vecs = vecs.T

for n in range(1, 4):
    # This is just A applied to x
    x = A.dot(x)

    # Here we compute the product via eigenvalues and x decomposition
    # into basis of eigenvectors
    x1 = vecs.dot(np.array([x_e0*vals[0]**n,
                            x_e1*vals[1]**n]))

    print x, x1



