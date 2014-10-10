
from dolfin import * 
import numpy 



def dump_matrix(filename, name, AA):
    f = open(filename, 'w')
    AA = AA.array()
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            if abs(AA[i,j]) > 10e-10: 
                f.write("%s (%d, %d) = %e;\n " % (name,i+1,j+1,AA[i,j])) 




# read input from command line
import sys
N = 4;  
for s in sys.argv[1:]:
  exec(s)


mesh = UnitSquareMesh(N, N)
# Create mesh and define function space
Vh = VectorFunctionSpace(mesh, "Lagrange", 2)
Ph = VectorFunctionSpace(mesh, "R", 0, 2)



(v, q) = TestFunction(Vh), TestFunction(Ph)
(u, p) = TrialFunction(Vh), TrialFunction(Ph)

aa =  inner(grad(u), grad(v))*dx 
bb = u[0]*q[0]*dx + u[1]*q[1]*dx 

pp =inner(u,v)*dx +  inner(grad(u), grad(v))*dx 
cc= p[0]*q[0]*dx + p[1]*q[1]*dx 


C = assemble(cc)
A = assemble(aa) 
B = assemble(bb)
P = assemble(pp)






dump_matrix("AA.m", "A", A)
dump_matrix("PP.m", "P", P)
dump_matrix("BB.m", "B", B)
dump_matrix("CC.m", "C", C)


import os
os.system("matlab -nodesktop < run.m")



