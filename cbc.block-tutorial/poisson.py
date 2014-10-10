from block import *
from block.iterative import *
from block.algebraic.petsc import *
from dolfin import *
from block.dolfin_util import *
import numpy

# Function spaces, elements

mesh = UnitSquareMesh(16, 16)

V = FunctionSpace(mesh, "CG", 1)

f = Expression('pi*pi*sin(pi*x[0]) + pi*pi*sin(pi*x[1])')
u, v = TrialFunction(V), TestFunction(V)

a = dot(grad(u), grad(v))*dx
L = f*v*dx

A, b = assemble_system(a, L)

# Create nullspace for dolfin
z = interpolate(Constant(1), V).vector()
normalize(z, 'l2')
z = [z]
nullspace = VectorSpaceBasis(z)

# Orthogonalize rhs
nullspace.orthogonalize(b)

# Set the nullspace for multigrid preconditioner
B = ML(A, nullspace=z)

Ainv = ConjGrad(A, precond=B, tolerance=1e-10, show=2)

x = Ainv*b

u = Function(V)
u.vector()[:] = x[:]
plot(u, title="u, computed by cbc.block [x=Ainv*b]")

u2 = Function(V)
solver = KrylovSolver('cg', 'amg')
solver.set_nullspace(nullspace)
solver.solve(A, u2.vector(), b)
plot(u2, title="u2, computed by dolfin [solve(A,x,b)]")

interactive()

