from dolfin import *
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as la


def dolfin_system(f, g, N, gamma_d):
    '''
    Assemble matrix of the Stokes system:

                    -laplace(u) + grad(p) = f
                                   div(u) = 0
                                u|gamma_d = g

    The problem is discretized on UnitSquareMesh(N, N).
    If gamma_d == 'everywhere' the Dirichlet condition on velocity is applied
    on the whole boundary - this introduces nullspace spanned by
    Constant(0, 0, 1).

    Retrun A sparse.array representing the system matrix
           u velocity solution
           p pressure solution
           eigs 7 smallest eigenvalues

    DOLFIN IS USED FOR ASSEMBLY.
    '''

    assert isinstance(f, cpp.function.GenericFunction)
    assert isinstance(g, cpp.function.GenericFunction)
    assert isinstance(N, int)
    assert isinstance(gamma_d, str)

    mesh = UnitSquareMesh(N, N)

    # Define variational problem
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    W = MixedFunctionSpace([V, Q])

    if gamma_d == 'everywhere':
        bc = DirichletBC(W.sub(0), g, 'on_boundary')
    else:
        bc = DirichletBC(W.sub(0), g,
                         lambda x, on_boundary: on_boundary and near(x[0], 0))

    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    a = inner(grad(u), grad(v))*dx - inner(div(v), p)*dx - inner(q, div(u))*dx
    L = inner(f, v)*dx

    A, b = PETScMatrix(), PETScVector()
    assemble_system(a, L, bc, A_tensor=A, b_tensor=b)

    # Solve the system to get idea about the solution
    uph = Function(W)
    solve(A, uph.vector(), b)

    uh, ph = uph.split(deepcopy=True)

    # Normalize the pressure (need for singular system)
    ph.vector()[:] -= assemble(ph*dx)

    # Check the eigenvalues:
    # Turn matrix to sparse scipy format
    A_ = csr_matrix(A.array())
    eigs, _ = la.eigs(A_, k=7, which='LM', sigma=1E-10)

    # For everywhere (0, 0, 1) should be in the nullspace
    Z = interpolate(Constant((0, 0, 1)), W).vector()
    null = PETScVector()
    A.mult(Z, null)

    # If the norm is very small the vector is in the nullspace, so there should
    # be eigenvalue 0
    print 'DOLFIN |A*(0, 0, 1)|_{l^2}', null.norm('l2')

    return (A_, uh, ph, eigs)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    f = Constant((1, 2))
    g = Constant((0, 0))
    dolfin_system(f=f, g=g, N=20, gamma_d='')
