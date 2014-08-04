clc
clear all
close all

load('test_data.mat');

[r c]= size(phi);
i=1;
k=1;
%w=pinv(phi)*t;
a=zeros(r,10)+0.01;
ep=0.1;
e=0;
c=0;
err=zeros(1,100);

%%%% finding softmax activation

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
dlmwrite('classes_lr.txt',y,'delimiter',' ');
err_rate=count/1500*100;
sprintf('The error rate for logistic regression %f percent.', err_rate)