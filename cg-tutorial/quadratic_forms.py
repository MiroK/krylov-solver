import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dolfin import BoxMesh, FunctionSpace, Function, File, plot

_TOL_ = 1E-15


def random_index(*index_sets):
    'Return index by selecting random element from index_sets.'
    index = []
    for index_set in index_sets:
        if isinstance(index_set, int):
            assert index_set > -1
            index.append(random.choice(range(index_set)))
        elif isinstance(index_set, (list, np.array)):
            index.append(random.choice(index_set))

    return tuple(index)


def visualize_form(A, b, c):
    '''
    Study the quadratic form phi(x) = 0.5*x*A*x - b*x + c
        - definiteness
        - extrema
        - constraints
    '''

    # Iput check
    assert isinstance(A, np.ndarray)
    assert A.shape == (2, 2)
    assert isinstance(b, np.ndarray)
    assert b.shape == (2, )
    assert isinstance(c, (float, int))

    # Decide symmetry
    is_sym = la.norm(A - A.T) < _TOL_
    is_skew = la.norm(A + A.T) < _TOL_

    print 'A is symmetric', is_sym, 'A is skew', is_skew

    # Decide definitness from eigenvalues
    vals, vecs = la.eig(A)
    vecs = vecs.T        # Eigenvectors are now row, easies for iter access
    print 'A eigenvalues', vals
    print 'A eigenvectors', vecs[0], vecs[1]

    is_singular, is_posdef, is_negdef, is_indef = False, False, False, False
    # Don't know about definiteness of forms from A with complex eigenvalues
    if not vals.dtype == 'complex128':
        if np.any(np.abs(vals) < _TOL_):
            is_singular = True
        else:
            if np.all(vals > 0):
                is_posdef = True
            elif np.all(vals < 0):
                is_negdef = True
            elif abs(np.sum(np.sign(vals))) < _TOL_:
                is_indef = True

        print 'A is singular', is_singular,\
              'A is posdef', is_posdef,\
              'A is negdef', is_negdef,\
              'A is indef', is_indef

    # lambda for computing the quadratic form value, and the gradient
    phi = lambda x: 0.5*x.dot(A.dot(x)) - b.dot(x) + c
    grad_phi = lambda x: A.dot(x) - b

    # Width for grids
    w = 3

    # If the system is not singular we can compute Ax=b (straight forward)
    # and use x to decide visualiz. grid width
    if not is_singular:
        if is_sym:
            x = la.solve(A, b)
        elif not is_skew:
            # If the system is not symmetric then the minima is sought
            # via the `symmetrized` system
            x = la.solve(0.5*(A + A.T), b)
        else:
            # For the skew system the above gives zero matrix and the gradient
            # is just b -- the quadratic surface is just a plane perpend to b
            x = b

    else:
        # For the singular system only constrained minimization makes sense
        # The functional is modified to include costraint that x is
        # perpendicular to nullspace vector

        # Get eigenvalues and eigenvectors of A
        # Find the eigenvector corresponding to 0. Is in the nullspace
        z = None
        for vec in vecs:
            if abs(vec.dot(A.dot(vec.T))) < _TOL_:
                z = vec

        # Build the system that is yields extrema of
        # phi(x) + constraint of orthogonality to z
        AA = np.zeros((3, 3))
        AA[:2, :2] = A
        AA[2, :2] = z
        AA[:2, 2] = z.T

        bb = np.zeros(3)
        bb[:2] = b

        xx = la.solve(AA, bb)

        # Extract x and the multiplier
        x = xx[:2]
        lmbda = xx[-1]
        print 'Multiplier', lmbda

        # lambda for computing surface whose contours are lines that satisfy
        # the constraint
        psi = lambda x: x.dot(z)

        # These are poits on the line that satisfies the constraint and the
        # solution passed through it
        cs = lambda s: x + np.array([z[1], -z[0]])*s

        # Chi is the constrained functional whose value we want to plot
        chi = lambda x: phi(x[:2]) + x[-1]*psi(x[:2])

        mesh = BoxMesh(x[0]-w, x[1]-w, lmbda-w, x[0]+w, x[1]+w, lmbda+w,
                       10, 10, 10)
        V = FunctionSpace(mesh, 'CG', 1)
        dofs_x = V.dofmap().tabulate_all_coordinates(mesh).reshape((-1, 3))

        values = np.array([chi(dof_x) for dof_x in dofs_x])

        # The plot should show that the solution has chosen a plane at certain
        # height = lagrange multiplier, that has normal = z
        chi_f = Function(V)
        chi_f.vector().set_local(values)
        chi_f.vector().apply('')
        plot(chi_f, interactive=True)

        print 'Saved constrained functional to chi.pvd'
        File('chi.pvd') << chi_f

        # Generate data for plot showing the quadratic function values over
        # cs line
        s_array = np.linspace(-2, 2, 1000)
        PHI_cs = np.zeros_like(s_array)
        for i, s in enumerate(s_array):
            P = cs(s)
            PHI_cs[i] = phi(P)

        # Add three contour plot x, y @ a*
        #                        x, a @ y* eval chi
        #                        y, a @ x*

    print 'Potential extremum', x

    X, Y = np.meshgrid(np.linspace(x[0]-w, x[0]+w, 100),
                       np.linspace(x[1]-w, x[1]+w, 100))

    PHI = np.zeros_like(X)
    GRAD_PHI = np.zeros((X.shape[0], X.shape[1], 2))

    # Generate values of qudratic function and its gradient
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            P = np.array([X[i, j], Y[i, j]])
            PHI[i, j] = phi(P)
            GRAD_PHI[i, j, :] = grad_phi(P)

    # Plot the quadratic form as 3d + the extreme point
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, PHI)
    ax.plot([x[0]], [x[1]], [phi(x)], 'rx')
    # ------------------------------------------------------------------------

    # Plot contours of the quadratic function
    plt.figure()
    plt.pcolor(X, Y, PHI)
    plt.contour(X, Y, PHI, 10, colors='k')
    plt.plot(x[0], x[1], 'rx')

    # Plot eigenvectors
    e0 = x + vecs[0]
    plt.plot(np.linspace(x[0], e0[0], 10), np.linspace(x[1], e0[1], 10), 'k')
    plt.text(0.5*x[0] + 0.5*e0[0], 0.5*x[1] + 0.5*e0[1], str(vals[0]))
    e1 = x + vecs[1]
    plt.plot(np.linspace(x[0], e1[0], 10), np.linspace(x[1], e1[1], 10), 'k')
    plt.text(0.5*x[0] + 0.5*e1[0], 0.5*x[1] + 0.5*e1[1], str(vals[1]))

    # For singular system add 'constraint lines'
    if is_singular:
        PSI = np.zeros_like(X)
        for i in range(PSI.shape[0]):
            for j in range(PSI.shape[1]):
                P = np.array([X[i, j], Y[i, j]])
                PSI[i, j] = psi(P)

        plt.contour(X, Y, PSI, 10, colors='g')
    # ------------------------------------------------------------------------

    # Finally plot curves tangent to gradient (better for  visualiz. than
    # quiver)
    plt.figure()
    plt.streamplot(X, Y, GRAD_PHI[:, :, 0], GRAD_PHI[:, :, 1])
    plt.plot(x[0], x[1], 'rx')
    if is_singular:
        plt.contour(X, Y, PSI, 10, colors='g')
    # ------------------------------------------------------------------------

    # Add the plot of phi(cs) to check that found point is the extremum
    if is_singular:
        plt.figure()
        plt.plot(s_array, PHI_cs, 'b')
        plt.plot([0], [phi(cs(0))], 'rx')

        plt.figure()
        plt.plot(s_array, np.array([chi(np.array([x[0], x[1], s])) for s
                                    in s_array]), label='via min')
        plt.plot(s_array, np.array([chi(np.array([x[0]+0.01, x[1]-0.01, s])) for s
                                    in s_array]), label='no min')
        plt.legend()
    # ------------------------------------------------------------------------

    plt.show()

    # For positve def and negdef forms see that we found minima and maxima
    # respectively
    # Something similar could be done for cs with singular system but the plot
    # tells the same story
    if is_posdef or is_negdef or is_indef:
        phi_s = np.array([PHI[random_index(PHI.shape[0], PHI.shape[1])]
                          for i in range(1000)])
        phi_s -= phi(x)

        if is_posdef:
            assert np.all(phi_s > 0)
        elif is_negdef:
            assert np.all(phi_s < 0)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    A_pd = np.array([[2, 0], [0, 3]])
    A_nd = -A_pd
    A_id = np.array([[1, 0], [0, -2]])
    A_skew = np.array([[0, 1], [-1, 0]])
    A_general = np.array([[3, 2], [2, 6]])
    A_0 = np.array([[1, 0], [0, 0]])

    b = np.array([2, -8])
    c = 0

    visualize_form(A_0, b, c)
