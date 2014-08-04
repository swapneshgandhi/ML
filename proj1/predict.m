%% Submitted by Swapnesh Gandhi Person #50096836
function [ output_args ] = predict(validation,validation_target,test,test_target,mean_f,var_t,w)

mc=10;      %defined Model complexity 10
lam=20;     %defined lambda=20

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Working on VALIDATION DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%replicating the input validation metrix mc times to create mc basis functions
validation=repmat(validation,1,mc);
[r,c]=size(validation);

%initialization
validation_mean=zeros(r,c);
exponent_v=zeros(r,c);
x_phi_v=zeros(r,c);

%replicated the mean and standard deviation variables mc times to create mc
%means and mc standard deviations
mean_f_v=repmat(mean_f,1,mc);
var_t_v=repmat(var_t,1,mc);

%creating error to be added to mean and standard deviation variables
var_m=0:(0.3/(46*mc)):0.3;
var_m=var_m(2:length(var_m));
var_s=0:(0.5/(46*mc)):0.5;
var_s=var_s(2:length(var_s));

% Adding the error to mean and standard deviation variables
var_t_v=var_t_v+var_s;
mean_f_v=mean_f_v+var_m;


%Replicated mean and standard deviation variables to match the validation matrix
%in terms of number of rows.
mean_f_v=repmat(mean_f_v,length(validation),1);
var_t_v=repmat(var_t_v,length(validation),1);

%Design Matrix phi for validation matrix
for i=1:r
    for j=1:c
       validation_mean(i,j)= validation(i,j)-mean_f_v(i,j);
       
       exponent_v(i,j)=validation_mean(i,j).^2;   
       if (var_t_v(i,j)~=0)
       exponent_v(i,j)=exponent_v(i,j)/(2*var_t_v(i,j));
       end
       x_phi_v(i,j)=exp(-1*exponent_v(i,j));
    end
end

%Predicting the target relevancy variables.
tar_main_v=x_phi_v*w;

%finding the error.
error=(tar_main_v-validation_target).^2;
err_sum=(sum(error))/2;

e_rms_va=sqrt(2*err_sum/r);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                      Working on TEST DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%replicating the input validation metrix mc times to create mc basis functions
test=repmat(test,1,mc);

% Initialization
[r,c]=size(test);
test_mean_t=zeros(r,c);
exponent_t=zeros(r,c);
x_phi_t=zeros(r,c);

%replicated the mean and standard deviation variables mc times to create mc
%means and mc standard deviations
mean_f_t=repmat(mean_f,1,mc);
var_t_t=repmat(var_t,1,mc);

%creating error to be added to mean and standard deviation variables
var_m=0:(0.3/(46*mc)):0.3;
var_m=var_m(2:length(var_m));
var_s=0:(0.5/(46*mc)):0.5;
var_s=var_s(2:length(var_s));
var_t_t=var_t_t+var_s;
mean_f_t=mean_f_t+var_m;


%Replicated mean and standard deviation variables to match the validation matrix
%in terms of number of rows.
mean_f_t=repmat(mean_f_t,length(test),1);
var_t_t=repmat(var_t_t,length(test),1);

%Design Matrix phi for Tetsing
for i=1:r
    for j=1:c
       test_mean_t(i,j)= test(i,j)-mean_f_t(i,j);
       exponent_t(i,j)=test_mean_t(i,j).^2;   
       if (var_t_t(i,j) ~= 0)
       exponent_t(i,j)=exponent_t(i,j)/(2*var_t_t(i,j));
       end
        x_phi_t(i,j)=exp(-1*exponent_t(i,j));
    end
end

%Predicting the target relevancy variables.
tar_main_t=x_phi_t*w;

%Finding Erms for testing
error=((tar_main_t-test_target)'*(tar_main_t-test_target))/2;
e_rms_ts=sqrt(2*error/r);


sprintf('the model complexity M for the linear regression model is %d', mc)
sprintf('the regularization parameters lambda for the linear regression model is %f', lam)
sprintf('the root mean square error for the linear regression model is %f', e_rms_ts)


end

