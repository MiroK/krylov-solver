from singular_problems import scalar_poisson_2d, vector_poisson_2d,\
    elasticity_2d
from scipy.sparse.linalg import eigs as sp_eigs
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from dolfin import assemble
import numpy as np
import os

__method__ = 'scipy'


def dump_matrix(filename, name, AA):
    with open(filename, 'w') as f:
        for i in range(AA.shape[0]):
            for j in range(AA.shape[1]):
                if abs(AA[i, j]) > 10e-10:
                    f.write("%s (%d, %d) = %e;\n " % (name, i+1, j+1, AA[i, j]))


def cond(A, M=None):
    'Get the condition number of A or inv(B)*A'
    if __method__ == 'scipy':
        large, _ = sp_eigs(A, k=2, M=M, which='LM')
        small, _ = sp_eigs(A, k=4, M=M, which='LM', sigma=1E-8, tol=1E-10)

        l_max = np.max(large)
        l_min = np.min(small)

        print 'Lambda_max', l_max, 'Lambda_min', l_min

        return abs(l_max)/abs(l_min)

    elif __method__ == 'matlab':
        # Dump the matrix to A.m
        dump_matrix('A.m', 'A', A)
        # Call matlab to get eigs for problem      B*u = a*u, stored in cond.m
        os.system('matlab -nodesktop < run_A.m')
        # Parse the file to get condition number
        with open('cond.m') as f:
            for line in f:
                print float((line.strip().split())[0])


def assemble_AA_BB(aa, bb, pp=None, cc=None):
    '''
    Return assembled system matrix [[aa, bb.T], [bb, 0]] and its preconditioner
    [[pp, 0], [cc]] from forms.
    '''
    # Get the blocks
    A = assemble(aa)
    B = assemble(bb)

    n = A.size(0)
    N = n + B.size(0)

    print 'System has size', N

    # Assemble system matrix and get its condition number
    AA = np.zeros((N, N))
    AA[:n, :n] = A.array()
    AA[:n, n:] = B.array().T
    AA[n:, :n] = B.array()
    AA = csr_matrix(AA)         # convert to sparse

    if pp is None and cc is None:
        return AA
    else:
        P = assemble(pp)
        C = assemble(cc)

        BB = np.zeros((N, N))
        BB[:n, :n] = P.array()
        BB[n:, n:] = C.array()
        BB = csr_matrix(BB)

        return AA, BB


def conds(AA, BB=None):
    'Return condition numbers of AA, BB and inv(BB)*AA'
    if BB is None:
        return cond(AA)
    else:
        return cond(AA), cond(BB), cond(AA, BB)


def test_optimality(problem, Ns):
    fig = plt.figure()
    ax = fig.gca()

    AA_conds = []
    BB_conds = []
    BB_AA_conds = []

    for N in Ns:
        forms = problem(N)
        matrices = assemble_AA_BB(*forms)
        AA_cond, BB_cond, BB_AA_cond = conds(*matrices)

        print N, AA_cond, BB_cond, BB_AA_cond

        AA_conds.append(AA_cond)
        BB_conds.append(BB_cond)
        BB_AA_conds.append(BB_AA_cond)

    ax.semilogy(Ns, AA_conds, '*-', label='$\kappa_A$')
    ax.semilogy(Ns, BB_conds, '*-', label='$\kappa_B$')
    ax.semilogy(Ns, BB_AA_conds, '*-', label='$\kappa_{B^{-1}A}$')
    ax.legend(loc='best')
    plt.show()

# test_optimality(vector_poisson_2d, [4, 8, 12, 14, 18])
# test_optimality(scalar_poisson_2d, range(4, 32))

forms = scalar_poisson_2d(40)
matrices = assemble_AA_BB(*forms)
foo = conds(*matrices)
print foo

# TODO tweak this to produce same results as matlab!
