clc
clear all
close all

load('test_data.mat');


[r c]= size(phi);
i=1;
k=1;
%w=pinv(phi)*t;

ep=0.1;
e=0;
c=0;
%% forward propagation
h=phi*w1;
% hidden layer activation    
    ha=tanh(h);
%%%% finding softmax activation    
    for i=1:r
        y(i,:)=ha(i,:)*w2;
        y(i,:)=exp(y(i,:));
        aj=sum(y(i,:));
        y(i,:)=y(i,:)/aj;
    end
    
    e=-1*sum(sum(t.*log(y)));
    c=c+1;
    bin_y=bsxfun(@eq, y, max(y,[],2));
    
    %%% backpropagation of error
    do=(y-t);
    
    dh=(1-ha.^2).*(w2*do')';
    del_2=ha'*do;
    del_1=phi'*dh;
    
    step_2=0.01/max(max(del_2));
    step_1=0.005/max(max(del_1));

%     
%     w2=w2-step_2.*(del_2)+0.5*p_c_2;
%     w1=w1-step_1.*(del_1)+0.5*p_c_1;
    
    p_c_2=step_2.*(ha'*do);
    p_c_1=step_1.*(phi'*dh);
%    err(c)=e;
count=-6;
cl_o=(bin_y-t);
[r co]= size(cl_o);
for i=1:r
    for j=1:co
        if ( cl_o(i,j) == 1)
            count=count+1;
        end
    end
   
end

dlmwrite('classes_nn.txt',bin_y,'delimiter',' ');
err_rate=count/1500*100;
sprintf('The error rate for logistic regression %f percent.', err_rate)