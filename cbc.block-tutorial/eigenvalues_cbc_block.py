from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as spla
import numpy.linalg as npla


def cbc_block_system(f, g, N, gamma_d):
    '''
    Assemble matrix of the Stokes system:

                    -laplace(u) + grad(p) = f
                                   div(u) = 0
                                u|gamma_d = g

    The problem is discretized on UnitSquareMesh(N, N).
    If gamma_d == 'everywhere' the Dirichlet condition on velocity is applied
    on the whole boundary - this introduces nullspace spanned by
    Constant(0, 0, 1).

    Retrun A the system matrix
           u velocity solution
           p pressure solution
           eigs 7 smallest eigenvalues

    CBC.BLOCK IS USED FOR ASSEMBLY.
    '''
    assert isinstance(f, cpp.function.GenericFunction)
    assert isinstance(g, cpp.function.GenericFunction)
    assert isinstance(N, int)
    assert isinstance(gamma_d, str)

    mesh = UnitSquareMesh(N, N)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)

    if gamma_d == 'everywhere':
        bc = DirichletBC(V, g, 'on_boundary')
    else:
        bc = DirichletBC(V, g,
                         lambda x, on_boundary: on_boundary and near(x[0], 0))

    # Define variational problem and assemble matrices
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)

    # Forms for blocks
    a11 = inner(grad(v), grad(u))*dx
    a12 = -div(v)*p*dx
    a21 = -div(u)*q*dx
    # NOTE: signs in a21 effect signs of eigenvalues!
    # This way the system is setup to get same eigs as DOLFIN counterpart
    L1 = inner(v, f)*dx

    # matrix is automatically created to replace the (2,2) block in AA,
    # since bc2 makes the block non-zero.
    bcs = [[bc], []]
    AA = block_assemble([[a11, a12],
                         [a21, 0]])
    bb = block_assemble([L1, 0])

    block_bc(bcs, True).apply(AA).apply(bb)

    # Extract the individual submatrices
    [[A, B],
     [C, _]] = AA

    [b, _] = bb

    # To solve turn to scipy and use iterative solver for comparison with
    # DOLFIN case
    A_ = np.zeros((A.size(0) + C.size(0), A.size(1) + B.size(1)))

    # Fill the blocks of mat
    A_[:A.size(0), :A.size(1)] = A.array()
    A_[:A.size(0), A.size(1):] = B.array()
    A_[A.size(0):, :A.size(1)] = C.array()
    # Fill the blocks of vec
    b_ = np.zeros(A.size(1) + B.size(1))
    b_[:A.size(1)] = b.array()

    A_ = csr_matrix(A_)
    up_ = spla.spsolve(A_, b_, use_umfpack=True)
    u_, p_ = up_[:A.size(1)], up_[A.size(1):]

    # Divide scipy solution for u, p
    u = Function(V)
    u.vector().set_local(u_)
    u.vector().apply('')

    p = Function(Q)
    p.vector().set_local(p_)
    p.vector().apply('')
    p.vector()[:] -= assemble(p*dx)  # Normalize

    # Check the eigenvalues
    eigs, _ = spla.eigs(A_, k=7, sigma=1E-10, which='LM')

    # For everywhere (0, 0, 1) should be in the nullspace
    z_ = np.zeros(A.size(1) + B.size(1))
    z_[A.size(1):] = interpolate(Constant(1), Q).vector().array()
    null_ = A_.dot(z_)

    # If the norm is very small the vector is in the nullspace, so there should
    # be eigenvalue 0
    print 'CBC.BLOCK |A*(0, 0, 1)|_{l^2}', npla.norm(null_)

    return (A_, u, p, eigs)

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    f = Constant((1, 2))
    g = Constant((0, 0))
    cbc_block_system(f=f, g=g, N=20, gamma_d='')
