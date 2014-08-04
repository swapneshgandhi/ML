clc
clear all 
close all
load('train_data.mat');
[r c]= size(phi);
i=1;
k=1;
%w=1-2.*rand(513,10);
a=zeros(r,10)+0.000000001;
ep=9.1;
e=9;
c=0;
err=zeros(1,100);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gradient descent loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%while( e> 10 )
    ep=-1*sum(sum(t.*log(a)));
    for i=1:r
    a(i,:)=phi(i,:)*w;
    a(i,:)=exp(a(i,:));
    aj=sum(a(i,:));
    a(i,:)=a(i,:)/aj;
    end
e=-1*sum(sum(t.*log(a)));
c=c+1;
y=bsxfun(@eq, a, max(a,[],2));

del_gra=phi'*(a-t);
step_2=0.05/max(max(del_gra));
    
w=w-step_2*(del_gra);
err_n(c)=e;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% finding error rate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count=0;
cl_o=(y-t);
[r c]= size(cl_o);
for i=1:r
    for j=1:c
        if ( cl_o(i,j) == 1)
            count=count+1;
            
        end
    end
    
end
%end