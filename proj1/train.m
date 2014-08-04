%% Submitted by Swapnesh Gandhi Person #50096836
function [ w ] = train( train_d,mean_f,var_t,train_target )

mc=10;      %defined Model complexity 10
lam=20;     %defined lambda=20

%replicating the metrix mc times to create mc basis functions
train_d=repmat(train_d,1,mc);   
[r,c]=size(train_d);

%Initialzed matrices
train_mean=zeros(r,c);      
exponent=zeros(r,c);
x_phi=zeros(r,c);
global w

%replicated the mean and standard deviation variables mc times to create mc
%means and mc standard deviations

mean_f=repmat(mean_f,1,mc);
var_t=repmat(var_t,1,mc);

%creating error to be added to mean and standard deviation variables
var_m=0:(0.3/(46*mc)):0.3;
var_m=var_m(2:length(var_m));
var_s=0:(0.5/(46*mc)):0.5;
var_s=var_s(2:length(var_s));

% Adding the error to mean and standard deviation variables
var_t=var_t+var_s;
mean_f=mean_f+var_m;

%Replicated mean and standard deviation variables to match the train matrix
%in terms of number of rows.
mean_f=repmat(mean_f,length(train_d),1);
var_t=repmat(var_t,length(train_d),1);

%creation of basis function phi
for i=1:r
    for j=1:c     
        
        train_mean(i,j)=train_d(i,j)-mean_f(i,j);
        exponent(i,j)=train_mean(i,j).^2;
         if (var_t(i,j)~=0)
        exponent(i,j)=exponent(i,j)/(2*var_t(i,j));
         end
        x_phi(i,j)=exp(-1*exponent(i,j));
    end
end

I=eye(c,c);
%Calculated the w using Maximum likelihood approach.
w=pinv((x_phi'*x_phi+lam*I))*x_phi'*train_target;
end
