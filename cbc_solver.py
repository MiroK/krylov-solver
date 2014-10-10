from forms import neumann_poisson_data, neumann_elasticity_data
from dolfin import *
from block import *
from block.iterative import *
from block.algebraic.petsc import *
from dolfin import *
from block.dolfin_util import *

# a, L, V, bc, z = neumann_poisson_data()
a, L, V, bc, z = neumann_elasticity_data()

A, b = PETScMatrix(), PETScVector()
assemble_system(a, L, A_tensor=A, b_tensor=b)

nullspace_basis = z
nullspace = VectorSpaceBasis(nullspace_basis)
nullspace.orthogonalize(b)

# Check basis properties
print 'Is orthogonal', nullspace.is_orthogonal()
print 'Is orthonormal', nullspace.is_orthonormal()

# Check b orthogonality
for zi in nullspace_basis:
    print '<zi, b> =', zi.inner(b)

# Myltigrid preconditioner
B = ML(A, nullspace=nullspace_basis)

Ainv = ConjGrad(A, tolerance=1e-10, show=2)

x = Ainv*b

u = Function(V)
U = u.vector()
U[:] = x[:]
plot(u, title='u')

# Check u orthogonality
for zi in nullspace_basis:
    print '<zi, U> =', zi.inner(U)

interactive()
