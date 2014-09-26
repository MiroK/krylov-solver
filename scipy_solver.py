from scipy.sparse import csr_matrix
import scipy.sparse.linalg as la
import numpy as np
import numpy.linalg as npla
from forms import neumann_poisson_data, neumann_elasticity_data
from dolfin import Matrix, Vector, Function, assemble_system, parameters,\
    plot, interactive
from functools import partial

np.set_printoptions(formatter={'all': lambda x: '%.5E' % x})

def copy_to_vector(U, U_array):
    assert U.size() == len(U_array)
    U.set_local(U_array)
    U.apply('')

# Get the forms
# a, L, V, bc, Z = neumann_poisson_data()

a, L, V, bc, Z = neumann_elasticity_data()

# Turn to dolfin.la objects
parameters.linear_algebra_backend = 'uBLAS'  # this one provides Matrix.data
A, b = Matrix(), Vector()
assemble_system(a, L, A_tensor=A, b_tensor=b)

# Turn to scipy objects
rows, cols, values = A.data()
AA = csr_matrix((values, cols, rows))

bb = np.array(b.array())

# Okay, first claim is that CG solver can't solve the problem Ax=b if the
# rhs is not perpendicular to the nullspace
ZZ = [np.array(Zi.array()) for Zi in Z]

for ZZi in ZZ:
    print '<b, Zi> =', ZZi.dot(bb)
x, info = la.cg(AA, bb)

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='original b')

# But can solve the problem once b is orthogonalized to the nullspace
for ZZi in ZZ:
    bb -= ZZi.dot(bb)*ZZi

for ZZi in ZZ:
    print '<b, Zi> =', ZZi.dot(bb)


# Start from 0 vector which of course orthogonal to Z
x, info = la.cg(AA, bb)

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='orthogonal b, x0=0')

# Start from x0 vector which has some component in Z
x0_in_Z = False
while not x0_in_Z:
    x0 = np.random.rand(len(b))
    for ZZi in ZZ:
        dot = ZZi.dot(x0)
        print dot
        if dot > 1E-15:
            x0_in_Z = True
            break
            # Found suitable x0

x, info = la.cg(AA, bb, x0=x0)

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='orthogonal b, <x0, Zi> != 0')

# Start from x0 vector which has some component in Z but correct
x0_in_Z = False
while not x0_in_Z:
    x0 = np.random.rand(len(b))
    for ZZi in ZZ:
        dot = ZZi.dot(x0)
        print dot
        if dot > 1E-15:
            x0_in_Z = True
            break
            # Found suitable x0


def orthogonalize(x, basis):
    for v in basis:
        x -= v.dot(x)*v

x, info = la.cg(AA, bb, x0=x0, callback=partial(orthogonalize, basis=ZZ))


uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='orthogonal b, <x0, Zi> != 0, callback')

# -----------------------------------------------------------------------------

# Check orthogonality of the solution
for ZZi in ZZ:
    print 'no projection <u, Z>', ZZi.dot(uh.vector().array())

# Projection to the nullspace
# my ZZ is such that row_i is ZZi, to be consisten paper where projection
# is Z*Z^T  ...
ZZZ = np.array(ZZ).T

ZZ_t = ZZZ.dot(ZZZ.T)

# A*(I - ZZ_t) and solve
AAP = AA - AA.dot(ZZ_t)
x, info = la.cg(AAP, bb)

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='AP')

for ZZi in ZZ:
    print 'APu = b: <u, Z>', ZZi.dot(uh.vector().array())

# (I - ZZ_t)*A and solve
PAA = AA - ZZ_t.dot(AA.todense())
x, info = la.cg(PAA, bb)

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='PA')

for ZZi in ZZ:
    print 'PAu = b: <u, Z>', ZZi.dot(uh.vector().array())

print 'PA nnz', np.count_nonzero(PAA), PAA.shape
print 'AP nnz', np.count_nonzero(AAP), AAP.shape
print 'A nnz', len(AA.nonzero()[0]), AA.shape

# Try with callback
class RangeProjector:
    def __init__(self, nullspace):
        self.n_iters = 0
        self.nullspace = nullspace

    def __call__(self, x):
        self.n_iters += 1

        # check orthogonality
        dots = np.array([zi.dot(x) for zi in self.nullspace])
        print self.n_iters, dots,

        # Orthogonalize
        for dot, zi in zip(dots, self.nullspace):
            x -= dot*zi

        # check orthogonality
        dots = np.array([zi.dot(x) for zi in self.nullspace])
        print dots

x, info = la.cg(AA, bb, callback=RangeProjector(ZZ))

uh = Function(V)
copy_to_vector(uh.vector(), x)
plot(uh, title='callback')

# How do PAA, AAP, A compare
D0 = PAA - AAP
print '|PAA-AAP|', npla.norm(D0)

D1 = PAA - AA
print '|PAA-AA|', npla.norm(D1)

D2 = AA - AAP
print '|AA-AAP|', npla.norm(D2)

interactive()
