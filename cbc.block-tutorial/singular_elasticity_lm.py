from dolfin import *
from block import *
from block.dolfin_util import *
from block.iterative import *
from block.algebraic.petsc import *
from rm_basis import eigen_basis

# Forces
f = Constant((0, 0, 0))
h = Constant(-2.5)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Strain-rate
epsilon = lambda u: sym(grad(u))

# Stress
sigma = lambda u: 2*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(3)

# -----------------------------------------------------------------------------

mesh = UnitCubeMesh(10, 10, 10)

# Define function spaces
V = VectorFunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(mesh, 'R', 0)
M = MixedFunctionSpace([R]*6)

# Define variational problem and assemble matrices
u, v = TrialFunction(V), TestFunction(V)
alphas, betas = TrialFunction(M), TestFunction(M)

# Get the basis for rigid motions
rm_basis = eigen_basis(mesh, ip='L2')

# Form for blocks of AA
a11 = inner(sigma(u), epsilon(v))*dx

a12, a21 = 0, 0
for i, e in enumerate(rm_basis):
    alpha = alphas[i]
    beta = betas[i]
    a12 += alpha*inner(v, e)*dx
    a21 += beta*inner(u, e)*dx

# a22 = 0

# Form for block of bb
boundaries = FacetFunction('size_t', mesh, 0)
AutoSubDomain(lambda x, on_boundary:\
              on_boundary and
              near(x[2]*(1 - x[2]), 0) and
              (between(x[0], (0.4, 0.6)) or between(x[1], (0.4, 0.6)))
              ).mark(boundaries, 1)

ds = Measure('ds')[boundaries]
n = FacetNormal(mesh)
L1  = inner(f, v)*dx + inner(h*n, v)*ds(1)

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
plot(Function(V, u), mode='displacement')
interactive()
