%--------------------------------------------------------------
%
%   Example of using LSTM regression to predict KoNViD-1k MOS
% 
%   Runs 100 times with different random splits to training
%   and test sets, stores result to file and displays average 
%   result
%

function results = predictMOSwithSVR_CrossDB(features_train, mos_train, ...
                                             features_test, mos_test)

seqlen=length(features_train);
XTrain = [];
for i=1:seqlen
    XTrain = [XTrain; mean(features_train{i})];
end
YTrain = mos_train;
seqlen=length(features_test);
XTest = [];
for i=1:seqlen
    XTest = [XTest; mean(features_test{i})];
end
YTest = mos_test;

% Train the SVR model
model = fitrsvm(XTrain, YTrain, ...
                'KernelFunction', 'gaussian', 'Standardize',true, ...
                'OptimizeHyperparameters', 'auto', 'Verbose', 0, ...
                'HyperparameterOptimizationOptions', ...
                struct('AcquisitionFunctionName',...
                       'expected-improvement-plus', ...
                       'MaxObjectiveEvaluations', 100, ...
                       'Verbose', 0, 'ShowPlots', false));

% Predict the values for the test set
YPred = predict(model, XTest);
results = [corr(YTest, YPred,'type','Pearson') ...
           corr(YTest, YPred,'type','Spearman') ...
           sqrt(mse(YTest, YPred))];

end

% EOF