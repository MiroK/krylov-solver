from rm_basis import eigen_basis as rigid_motions
from dolfin import *


def scalar_poisson_2d(N):
    '''
    Return forms for system matrix [[aa, bb.T], [bb, 0]] and its
    preconditioner [[pp, 0], [cc]] of the scalar Poisson problem on
    UnitSquareMesh(N, N)
    '''
    # Create mesh and define function spaces
    mesh = UnitSquareMesh(N, N)
    Vh = FunctionSpace(mesh, 'CG', 1)
    Ph = FunctionSpace(mesh, 'R', 0)

    u, v = TrialFunction(Vh), TestFunction(Vh)
    p, q = TrialFunction(Ph), TestFunction(Ph)

    # A[0, 0] block form
    aa = inner(grad(u), grad(v))*dx

    # A[1, 0] block form
    bb = inner(u, q)*dx

    # Preconditioner B, B[0, 0] block form
    pp = inner(u, v)*dx + inner(grad(u), grad(v))*dx
    # B[1, 1] form
    cc = inner(p, q)*dx

    return aa, bb, pp, cc


def vector_poisson_2d(N):
    '''
    Return forms for system matrix [[aa, bb.T], [bb, 0]] and its
    preconditioner [[pp, 0], [cc]] of the vector Poisson problem on
    UnitSquareMesh(N, N)
    '''
    # Create mesh and define function spaces
    mesh = UnitSquareMesh(N, N)
    Vh = VectorFunctionSpace(mesh, 'CG', 2)
    Ph = VectorFunctionSpace(mesh, 'R', 0, 2)

    u, v = TrialFunction(Vh), TestFunction(Vh)
    p, q = TrialFunction(Ph), TestFunction(Ph)

    # A[0, 0] block form
    aa = inner(grad(u), grad(v))*dx

    # A[1, 0] block form
    bb = u[0]*q[0]*dx + u[1]*q[1]*dx

    # Preconditioner B, B[0, 0] block form
    pp = inner(u, v)*dx + inner(grad(u), grad(v))*dx
    # B[1, 1] form
    cc = p[0]*q[0]*dx + p[1]*q[1]*dx

    return aa, bb, pp, cc


def elasticity_2d(N):
    '''
    Return forms for system matrix [[aa, bb.T], [bb, 0]] and its
    preconditioner [[pp, 0], [cc]] of the elasticity problem on
    UnitSquareMesh(N, N)
    '''
    # Elasticity parameters
    E, nu = 10.0, 0.3
    mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    # Strain-rate
    epsilon = lambda u: sym(grad(u))

    # Stress
    sigma = lambda u: 2*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(2)

    # Create mesh and define function spaces
    mesh = UnitSquareMesh(N, N)
    Vh = VectorFunctionSpace(mesh, 'CG', 1)
    Ph = VectorFunctionSpace(mesh, 'R', 0, 3)

    u, v = TrialFunction(Vh), TestFunction(Vh)
    p, q = TrialFunction(Ph), TestFunction(Ph)

    # A[0, 0] block form
    aa = inner(sigma(u), epsilon(v))*dx

    # A[1, 0] block form
    rms = rigid_motions(mesh, ip='L2')
    bb = sum([q[i]*inner(rms[i], u)*dx for i in range(len(rms))])

    # Preconditioner B, B[0, 0] block form
    pp = None #inner(u, v)*dx + inner(eps(u), eps(v))*dx
    # B[1, 1] form
    cc = None #p[0]*q[0]*dx + p[1]*q[1]*dx

    return aa, bb, pp, cc
