% Solve a Pattern Recognition Problem with a Neural Network
% Script generated by NPRTOOL
% Created Tue Dec 03 11:30:52 EST 2013
%
% This script assumes these variables are defined:
%
%   phi - input data.
%   t - target data.

inputs = phi';
targets = t';

% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);


% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 5/100;
net.divideParam.testRatio = 20/100;


% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)
bin_y=bsxfun(@eq, outputs(16000:19978), max(outputs(16000:19978)));
count=0;
cl_o=target(16000:19978)-bin_y;
[r c]= size(cl_o);
for i=1:r
    for j=1:c
        if ( cl_o(i,j) == 1)
            count=count+1;
            
        end
    end
    
end


err_rate=count/1500*100;
sprintf('The error rate for logistic regression %f percent.', err_rate)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(targets,outputs)
%figure, ploterrhist(errors)
