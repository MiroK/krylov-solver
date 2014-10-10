from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *


u_exact = Expression('sin(pi*x[0]) + sin(pi*x[1]) + \
                     pi*(x[0]*(x[0]-1) + x[1]*(x[1]-1)) - 4/pi + pi/3')

f = Expression('pi*pi*sin(pi*x[0]) + pi*pi*sin(pi*x[1])')

# -----------------------------------------------------------------------------

mesh = UnitSquareMesh(40, 40)

# Define function spaces
V = FunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(mesh, 'R', 0)

# Define variational problem and assemble matrices
u, v = TrialFunction(V), TestFunction(V)
lmbda, beta = TrialFunction(R), TestFunction(R)

# Form for blocks of AA
a11 = inner(grad(v), grad(u))*dx
a12 = inner(lmbda, v)*dx
a21 = inner(u, beta)*dx
# a22 = 0

# Form for block of bb
L1  = inner(v, f)*dx

# Assemble the matrix
AA = block_assemble([[a11, a12],
                     [a21,  0 ]])
bb  = block_assemble([L1, 0])

# Extract the individual submatrices
[[A, B],
 [C, _]] = AA

# Create the block inverse
AAinv = MinRes(AA, tolerance=1e-10, maxiter=500, show=2)

# Compute solution
u, p = AAinv * bb

# Plot solution
plot(Function(V, u))
plot(u_exact, mesh)
interactive()
