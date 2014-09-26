from dolfin import UnitSquareMesh, FunctionSpace, TrialFunction, TestFunction,\
    Expression, inner, DirichletBC, Constant, DomainBoundary, dx, grad,\
    interpolate, normalize, VectorFunctionSpace, tr, Identity, sym
import numpy.linalg as la
import numpy as np


def neumann_poisson_data():
    '''
    Return:
        a  bilinear form in the neumann poisson problem
        L  linear form in therein
        V  function space, where a, L are defined
        bc homog. dirichlet conditions for case where we want pos. def problem
        z  list of orthonormal vectors in the nullspace of A that form basis
           of ker(A)

    '''
    mesh = UnitSquareMesh(40, 40)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression('x[0]+x[1]')
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    bc = DirichletBC(V, Constant(0), DomainBoundary())

    z0 = interpolate(Constant(1), V).vector()
    normalize(z0, 'l2')

    print '%16E' % z0.norm('l2')
    assert abs(z0.norm('l2')-1) < 1E-13

    return a, L, V, bc, [z0]


def neumann_elasticity_data():
    '''
    Return:
        a  bilinear form in the neumann elasticity problem
        L  linear form in therein
        V  function space, where a, L are defined
        bc homog. dirichlet conditions for case where we want pos. def problem
        z  list of orthonormal vectors in the nullspace of A that form basis
           of ker(A)
    '''
    mesh = UnitSquareMesh(40, 40)

    V = VectorFunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression(('sin(pi*x[0])', 'cos(pi*x[1])'))

    epsilon = lambda u: sym(grad(u))

    # Material properties
    E, nu = 10.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    sigma = lambda u: 2*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(2)

    a = inner(sigma(u), epsilon(v))*dx
    L = inner(f, v)*dx  # Zero stress

    bc = DirichletBC(V, Constant((0, 0)), DomainBoundary())

    z0 = interpolate(Constant((1, 0)), V).vector()
    normalize(z0, 'l2')

    z1 = interpolate(Constant((0, 1)), V).vector()
    normalize(z1, 'l2')

    X = mesh.coordinates().reshape((-1, 2))
    c0, c1 = np.sum(X, axis=0)/len(X)
    z2 = interpolate(Expression(('x[1]-c1',
                                 '-(x[0]-c0)'), c0=c0, c1=c1), V).vector()
    normalize(z2, 'l2')

    z = [z0, z1, z2]

    # Check that this is orthonormal basis
    I = np.zeros((3, 3))
    for i, zi in enumerate(z):
        for j, zj in enumerate(z):
            I[i, j] = zi.inner(zj)

    print I
    print la.norm(I-np.eye(3))
    assert la.norm(I-np.eye(3)) < 1E-13

    return a, L, V, bc, z

if __name__ == '__main__':
    neumann_poisson_data()

    print

    neumann_elasticity_data()
