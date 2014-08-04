clc
clear all
load('train_data.mat');
[r c]= size(phi);

% w1=1-2.*rand(513,333);
% w2=1-2.*rand(333,10);
ep=0.1;
e=0;
c=0;
y=zeros(r,10)+0.01;
    

p_c_2=0;
p_c_1=0;
count=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% gradient descent loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%while ( count > 47 )
    ep=-1*sum(sum(t.*log(y)));
    h=phi*w1;
    
    ha=tanh(h);
    
    for i=1:r
        y(i,:)=ha(i,:)*w2;
        y(i,:)=exp(y(i,:));
        aj=sum(y(i,:));
        y(i,:)=y(i,:)/aj;
    end
    
    e=-1*sum(sum(t.*log(y)));
    c=c+1;
    bin_y=bsxfun(@eq, y, max(y,[],2));
    
    do=(y-t);
    
    dh=(1-ha.^2).*(w2*do')';
    del_2=ha'*do;
    del_1=phi'*dh;
    
    step_2=0.05/max(max(del_2));
    step_1=0.001/max(max(del_1));

%     
%     w2=w2-step_2.*(del_2)+0.5*p_c_2;
%     w1=w1-step_1.*(del_1)+0.5*p_c_1;
    
    p_c_2=step_2.*(ha'*do);
    p_c_1=step_1.*(phi'*dh);
    err(c)=e;

%end

count=0;
cl_o=(bin_y-t);
[r c]= size(cl_o);
for i=1:r
    for j=1:c
        if ( cl_o(i,j) == 1)
            count=count+1;
            
        end
    end
    
end


