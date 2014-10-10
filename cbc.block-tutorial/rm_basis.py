from dolfin import *
import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

def plot_axis(fig, mesh, axis, gc, **kwargs):

    assert mesh.topology().dim() == 3
    mesh.init(2, 3)
    mesh.init(1, 2)
    mesh.init(1, 0)

    bdry_edges = (edge for edge in edges(mesh)
                  if any(Facet(mesh, index).exterior()
                         for index in edge.entities(2)))

    vertices = mesh.coordinates().reshape((mesh.num_vertices(), 3))

    ax = fig.gca(projection='3d')

    # Plot the mesh surface
    plot_mesh = False
    if plot_mesh:
        for edge in bdry_edges:
            v0, v1 = edge.entities(0)
            V0, V1 = vertices[v0], vertices[v1]
            ax.plot([V0[0], V1[0]], [V0[1], V1[1]], [V0[2], V1[2]], 'k')

    l = np.min([np.max(vertices[:, i])-np.min(vertices[:, i])
                for i in range(3)])

    for vec in axis:
        end_point = l*vec + gc
        ax.plot([gc[0], end_point[0]],
                [gc[1], end_point[1]],
                [gc[2], end_point[2]], kwargs.get('color'))


def eigen_basis(mesh, ip, full_output=False):
    '''
    Return basis of the space of rigid motions over mesh.
    The basis is constructed from eigenvectors of `geometrical` inertia
    tensor.

    full_output True will return the computed eigenvectors and geometric center
    '''
    gdim = mesh.geometry().dim()
    tdim = mesh.topology().dim()

    assert tdim > 1 and gdim > 1
    assert gdim == tdim

    # Define quantities based on the inner product
    if ip == 'L2':
        # Volume measure of the entire mesh
        dV = Measure('dx', domain=mesh)
        # Get the volume
        volume = assemble(Constant(1.)*dV)
        # Compute geometrix center
        gc = [assemble(Expression('x[i]', i=i)*dV)/volume for i in range(gdim)]
        # Define how to compute entries of the geometric tensor
        assemble_tensor_value = lambda u, v: assemble(inner(u, v)*dV)
        # Define how to manipute eigenvectors of the geometric tensor
        get_basis_vector = lambda u: u

    else:
        # We need scalar space for computing the center (bit simpler than
        # dofmaps)
        S = FunctionSpace(mesh, 'CG', 1)
        # Volume is just a number of vertices
        volume = S.dim()
        # Sum x_i components of vertices
        gc = [interpolate(Expression('x[i]', i=i), S).vector().sum()/volume
              for i in range(gdim)]

        V = VectorFunctionSpace(mesh, 'CG', 1)
        # Tensor components are computed by L2 inner products
        assemble_tensor_value = lambda u, v:\
            interpolate(u, V).vector().inner(interpolate(v, V).vector())
        # We extract vector of expansion coefficients
        get_basis_vector = lambda u: interpolate(u, V).vector()

    # Only 3d case need the tensor computations
    if gdim == 3:
        # Define expressions for tensor computations
        e0 = Expression(('0', '-(x[2] - Z)', 'x[1] - Y'), Y=gc[1], Z=gc[2])
        e1 = Expression(('x[2] - Z', '0', '-(x[0] - X)'), X=gc[0], Z=gc[2])
        e2 = Expression(('-(x[1] - Y)', 'x[0] - X', '0'), X=gc[0], Y=gc[1])
        es = [e0, e1, e2]

        # Assemble the intertia tensor
        I = np.zeros((3, 3))
        for i, ei in enumerate(es):
            for j, ej in enumerate(es):
                I[i, j] = assemble_tensor_value(ei, ej)

        # Compute the eigenvectors of I
        lambda_s, us = la.eigh(I)
        us = us.T     # Now rows are the eigenvectors

        # Get basis for translations by properly normalizing the eigenvectors
        translation = [get_basis_vector(Constant(tuple(us[:, i]*sqrt(volume)**-1)))
                    for i in range(len(us))]

        # Get basis for rotation
        # Here we transoform cannonical rotations via eigenvectors and normalize
        # with eigenvalues
        rotation = [get_basis_vector(Expression(('A*(v*(x[2]-Z) - w*(x[1]-Y))',
                                                'A*(w*(x[0]-X) - u*(x[2]-Z))',
                                                'A*(u*(x[1]-Y) - v*(x[0]-X))'),
                                                u=u, v=v, w=w,
                                                X=gc[0], Y=gc[1], Z=gc[2], A=A))
                    for (u, v, w), A in zip(us, np.sqrt(1./lambda_s))]
    elif gdim == 2:
        # Normalize axes for translational displacements
        ui = 1./sqrt(volume)
        translation = [get_basis_vector(u)
                       for u in [Constant((ui, 0)), Constant((0, ui))]]

        rotation = Expression(('-A*(x[1] - Y)', 'A*(x[0] - X)'),
                              X=gc[0], Y=gc[1], A=1)
        A = assemble_tensor_value(rotation, rotation)
        rotation.A = 1./sqrt(A)
        rotation = [get_basis_vector(rotation)]

        # The body axis of rotation is (0, 0, 1)
        us = np.array([[0, 0, 1]])

    # Rigid displacements
    rm_basis = translation + rotation

    if full_output:
        return rm_basis, (us, gc)
    else:
        return rm_basis

if __name__ == '__main__':
    # mesh = Mesh('../meshes/femur.xml')
    # mesh = Mesh('../meshes/sphere_sphere.xml')
    mesh = Mesh('../meshes/cube.xml')
    # mesh = UnitCubeMesh(10, 10, 10)

    #mesh = UnitSquareMesh(20, 20)

    fig = plt.figure()

    for ip, color in zip(['L2', 'l2'], ['r', 'b']):
        rm_basis, (axis, center) = eigen_basis(mesh, ip=ip, full_output=True)

        # Chuck that the basis is really orthonormal in the proper inner product
        dim_rm = len(rm_basis)
        M = np.zeros((dim_rm, dim_rm))
        for i, u in enumerate(rm_basis):
            for j, v in zip(range(i, dim_rm), rm_basis[i:]):
                M[i, j] =\
                    u.inner(v) if ip == 'l2' else assemble(inner(u, v)*dx(domain=mesh))
                M[j, i] = M[i, j]

        print 'basis matrix %s\n' % ip, M

        plot_axis(fig, mesh, axis, center, color=color)

    ax = fig.gca(projection='3d').set_axes('equal')
    plt.show()

