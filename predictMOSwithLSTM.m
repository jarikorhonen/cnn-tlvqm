%---------------------------------------------------------------
%
%   Example of using LSTM regression to predict KoNViD-1k MOS
%
%   Written by Jari Korhonen, Shenzhen University
%
%   Runs 100 times with different random splits to training
%   and test sets, stores result to file and displays average 
%   result
%

feature_file = '.\\KONVID_features.mat';
result_file = '.\\KONVID_LSTM_result.csv';

% Load data (features and respective MOS) and initialize variables
load(feature_file);
mos = (mos-1)./4; % Normalize MOS
seqlen=length(features);
for i=1:seqlen
    features{i}=features{i}';
end
results = [];

% Compute results for 100 random splits
for i=1:100

    % Initialize random number generation
    rng(2*i);      
    rand_seq = randperm(seqlen);

    % Split data to training and test sets    
    XTrain = features(rand_seq(1:ceil(0.8*seqlen)));
    YTrain = mos(rand_seq(1:ceil(0.8*seqlen)));
    XTest = features(rand_seq(ceil(0.8*seqlen)+1:seqlen));
    YTest = mos(rand_seq(ceil(0.8*seqlen)+1:seqlen));

    % Define LSTM network
    numResponses = 1;
    numFeatures = size(XTrain{1},1);
    layers = [ sequenceInputLayer(numFeatures)
               lstmLayer(512,'OutputMode','last')
               fullyConnectedLayer(numResponses)
               huberRegressionLayer('Huber')];

    % Define learning parameters
    maxEpochs = 50;
    miniBatchSize = 32;
    options = trainingOptions('sgdm', ...
                              'MaxEpochs',maxEpochs, ...
                              'MiniBatchSize',miniBatchSize, ...
                              'LearnRateSchedule','piecewise', ...
                              'LearnRateDropFactor',0.2, ...
                              'LearnRateDropPeriod',10, ...
                              'InitialLearnRate',0.02, ...
                              'L2Regularization',0.001, ...
                              'ExecutionEnvironment','cpu', ...
                              'Shuffle','every-epoch', ...
                              'ValidationData',{XTest,YTest}, ...
                              'ValidationFrequency',100, ...
                              'Plots','none','Verbose',0);

    % Train network
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Predict the values for the test set
    YPred = predict(net,XTest,'ExecutionEnvironment','cpu')';
    new_result = [corr(YTest, YPred','type','Spearman') ...
                  corr(YTest, YPred','type','Pearson') ...
                  sqrt(mse(YTest, YPred'))];

    % Store and display results for this round
    results = [results; new_result];
    fprintf('Round %d: SRCC %2.3f PLCC %2.3f RMSE %0.4f\n', i, ...
                new_result(1),new_result(2),new_result(3));              
end

% Compute and display the final results
means = mean(results);
stds = std(results);
fprintf('Total results: ');
fprintf('SRCC %2.3f (%1.3f) PLCC %2.3f (%1.3f) RMSE %0.4f (%0.4f)\n', ...
            means(1),stds(1),means(2),stds(2),means(3),stds(3));
csvwrite(result_file,results);

% EOF