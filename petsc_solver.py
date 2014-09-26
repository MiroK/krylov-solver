from forms import neumann_poisson_data, neumann_elasticity_data
from dolfin import *
import numpy as np

def copy_to_vector(U, U_array):
    assert U.size() == len(U_array)
    U.set_local(U_array)
    U.apply('')

# a, L, V, bc, z = neumann_poisson_data()

a, L, V, bc, z = neumann_elasticity_data()

A, b = PETScMatrix(), PETScVector()
assemble_system(a, L, A_tensor=A, b_tensor=b)

# z0 = interpolate(Expression('sin(pi*x[0])'), V).vector()
# normalize(z0, 'l2')    # Don't forget, vectorspacebasis is not # normalizing

# z1 = interpolate(Expression('cos(pi*x[1])'), V).vector()
# normalize(z1, 'l2')    # Don't forget, vectorspacebasis is not # normalizing

# print z0.inner(z1)

# nullspace_basis = [z0, z1]
# nullspace = VectorSpaceBasis(nullspace_basis)

nullspace_basis = z
nullspace = VectorSpaceBasis(nullspace_basis)

# Othogonality before orthogonalization
for zi in nullspace_basis:
    print '<b, zi> before', zi.inner(b)

nullspace.orthogonalize(b)

# Othogonality after orthogonalization
for zi in nullspace_basis:
    print '<b, zi> after', zi.inner(b)

for attach_nullspace in [True, False]:
    for x0_in_Z in [True, False]:
        if attach_nullspace:
            P = PETScPreconditioner('amg')
            P.set_nullspace(nullspace)
            solver = PETScKrylovSolver('cg', P)
            solver.set_nullspace(nullspace)
        else:
            solver = KrylovSolver('cg', 'amg')

        solver.set_operator(A)

        solver.parameters['maximum_iterations'] = 500
        solver.parameters['error_on_nonconvergence'] = False

        u = Function(V)
        U = u.vector()
        if x0_in_Z:
            solver.parameters['nonzero_initial_guess'] = True
            # Create vector which has component in Z
            FLAG = False
            while not FLAG:
                U_array = np.random.rand(b.size())
                for zi in nullspace_basis:
                    dot = zi.array().dot(U_array)
                    print dot
                    if dot > 1E-15:
                        FLAG = True
                        break
            copy_to_vector(U, U_array)
        else:
            pass
            # has solver.parameters['nonzero_initial_guess'] = False and 0 is okay

        solver.solve(U, b)

        # Check solutions orthogonality
        print 'attached=%s, x0_in_Z=%s' % (attach_nullspace, x0_in_Z)
        for zi in nullspace_basis:
            print '<u, zi> orthogonality', zi.inner(U)

        plot(u, title='Attached nullspace %s, x0 in Z %s' % (attach_nullspace,
                                                            x0_in_Z))
interactive()


# ----------------------------------------------------------------------------
# Let's do things by hand
AA, bb, bb_og = PETScMatrix(), PETScVector(), PETScVector()
# Newly assembled matrix, vector
assemble_system(a, L, A_tensor=AA, b_tensor=bb)
assemble(L, tensor=bb_og)

# Orthogonalize manually, version with extra vector
[bb_og.axpy(-1, zi.inner(bb)*zi) for zi in nullspace_basis]

# Othogonality after orthogonalization
for zi in nullspace_basis:
    print '<b, zi> after, manual extra', zi.inner(bb_og)

# Orthogonalize manually, version with extra vector
[b.axpy(-1, zi.inner(b)*zi) for zi in nullspace_basis]

# Othogonality after orthogonalization
for zi in nullspace_basis:
    print '<b, zi> after, manual in place', zi.inner(bb_og)
