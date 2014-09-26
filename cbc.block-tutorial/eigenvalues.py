'''
Assemble the Stokes system with DOLFIN and with cbc.block.
Check that the solutions and eigenvalues match but structure of matrices
is different.
'''

from dolfin import Expression, Constant, UnitSquareMesh, assemble, inner, dx
from eigenvalues_dolfin import dolfin_system
from eigenvalues_cbc_block import cbc_block_system
import matplotlib.pyplot as plt

# Right hand side and boundary value for velocity
f = Expression(('sin(pi*x[1])', 'cos(pi*(x[0]+x[1]))'))
g = Constant((0, 0))

N = 20
gamma_d = 'everywhere'
Ad, ud, pd, eigs_d = dolfin_system(f=f, g=g, N=N, gamma_d=gamma_d)
Ac, uc, pc, eigs_c = cbc_block_system(f=f, g=g, N=N, gamma_d=gamma_d)

plt.figure()
plt.suptitle('DOLFIN')
plt.spy(Ad, marker='.')

plt.figure()
plt.suptitle('CBC.BLOCK')
plt.spy(Ac, marker='.')
plt.show()

mesh = UnitSquareMesh(20, 20)
print 'Velocity diff', assemble(inner(ud-uc, ud-uc)*dx(domain=mesh))
print 'Pressure diff', assemble(inner(pd-pc, pd-pc)*dx(domain=mesh))

print 'DOLFIN eigs', eigs_d
print 'CBC.BLOCK eigs', eigs_c
print 'Norm of diff', (eigs_d - eigs_c).dot(eigs_d - eigs_c)


# All okay
# Setting sigma for eigensolver to detect zero!
# Sign of a21 to get eigenvalue agreement with DOLFIN
