import numpy as np
import numpy.linalg as la

def solve(A, b, x0, eps=1E-8, iter_max=1000):
    'Solve system Ax=b with Jacobi method.'

    # Checks of input
    m, n = A.shape
    assert m == n
    assert b.shape == (n, )

    # To prevent some integer division surprises with inverse
    A = A.astype('float', copy=False)
    b = b.astype('float', copy=False)

    # Get Jacobi matrix
    # If the diagonal is all zero, there is no way to continues
    if la.norm(A.diagonal()) < 1E-15:
        print 'Is diag(A) all zeros?'
        return x0
    else:
        diag_inv = np.array([1./d if abs(d) > 1E-14 else 0
                             for d in A.diagonal()])
        D_inv = np.diag(diag_inv)

    E = A - np.diag(A.diagonal())
    if la.norm(E) < 1E-13:
        E = np.eye(n)

    B = -D_inv.dot(E)

    # compute the rhs constant term
    z = D_inv.dot(b)

    n_iters = 0
    r = b - A.dot(x0)       # Initialize residuum
    r_norm = r.dot(r)
    r_norm0 = r_norm        # Remember size for stopping loop

    # Compare the spectra
    A_vals = la.eigvals(A)
    B_vals = la.eigvals(B)

    print 'A eigenvalues', A_vals
    print 'B eigenvalues', B_vals

    if not np.all(np.abs(B_vals) < 1):
        print '\tSome eigenvalues of B are greate than one in magnitude!'

    while n_iters < iter_max and r_norm > eps*r_norm0:
        n_iters += 1

        # Get the new solution
        x0 = B.dot(x0) + z

        # Compute norm of residuum
        r = b - A.dot(x0)
        r_norm = r.dot(r)

    return x0, n_iters

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    A_pd = np.array([[2, 0], [0, 3]])
    A_nd = -A_pd
    A_id = np.array([[1, 0], [0, -2]])
    A_idd = np.array([[1, 2], [1, -2]])
    A_skew = np.array([[0, 1], [-1, 0]])
    A_general = np.array([[3, 2], [2, 6]])
    A_0 = np.array([[1, 0], [0, 0]])

    A = A_idd
    b = np.array([2, -8])
    x0 = np.zeros(2)

    x0, n_iters = solve(A, b, x0)
    print 'jacobi finished in %d iterations' % n_iters, x0

    try:
        x_np = la.solve(A, b)
        print 'np sol', x_np, 'diff', la.norm(x_np - x0)
    except la.LinAlgError:
        print 'Singular matrix'

    # Jacobi does not work with skew matrices - they have 0 on diagonal
    # Indefinite systems work if they are only A ~ diagonal
    # rho(B) might inlude > 1 but this is not a problem as long as you don't
    # enter iteration loop, them blow up
