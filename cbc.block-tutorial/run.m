
AA; 
BB; 
CC; 
PP; 

AA = [A, B'; B, 0*C]; 
BB = [P, 0*B'; 0*B, C]; 

e = eig(AA); 

e(1);
n=size(e);
condition_number = e(n(1)) / e(1)

e = eig(BB); 

e(1);
n=size(e);
condition_number = e(n(1)) / e(1)




e = sort(abs(eig(AA, BB))); 

e(1);
n=size(e);
condition_number = e(n(1)) / e(1)

