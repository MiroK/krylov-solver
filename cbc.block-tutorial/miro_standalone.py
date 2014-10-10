from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
from singular_problems import *
import matplotlib.pyplot as plt
from dolfin import assemble
import numpy as np


def cond(A, M=None):
    'Get the condition number of A or inv(B)*A'
    large, _ = eigs(A, k=2, M=M, which='LM')
    small, _ = eigs(A, k=2, M=M, which='LM', sigma=1E-10)

    l_max = np.max(large)
    l_min = np.min(small)

    print 'Lambda_max', l_max, 'Lambda_min', l_min

    return abs(l_max)/abs(l_min)


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
    AA = csr_matrix(AA)

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
test_optimality(scalar_poisson_2d, range(4, 32))

# forms = elasticity_2d(4)
# matrices = assemble_AA_BB(*forms)
# AA_cond = conds(matrices)
# print AA_cond

# TODO tweak this to produce same results as matlab!
