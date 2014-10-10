A; 

e = eig(A); 

e(1);
n=size(e);
condition_number = e(n(1)) / e(1);

save('cond.m', 'condition_number', '-ASCII')

