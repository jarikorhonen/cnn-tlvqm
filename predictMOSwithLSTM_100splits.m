%--------------------------------------------------------------
%
%   Example of using LSTM regression to predict KoNViD-1k MOS
% 
%   Runs 100 times with different random splits to training
%   and test sets, stores result to file and displays average 
%   result
%

function results = predictMOSwithLSTM_100splits(features, mos)

seqlen=length(features);
for i=1:seqlen
    features{i}=features{i}';
end
results = [];
new_result = [0 0 0];
indicator_text = '';

% Compute results for 100 random splits
for i=1:100

    % Initialize random number generation
    rng(2*i);      
    rand_seq = randperm(seqlen);
    
    fprintf(repmat(char(8), 1, length(indicator_text)));
    indicator_text = sprintf('Training and testing split %d/100\n',i);
    if i>2
        means = mean(results);
        indicator_text = [sprintf(...
            'Average results after round %d: PCC %2.3f SCC %2.3f RMSE %0.4f\n', ...
             i-1,means(1),means(2),means(3)) ...
             indicator_text];
    end
    fprintf(indicator_text);

    % Split data to training and test sets    
    XTrain = features(rand_seq(1:ceil(0.8*seqlen)));
    YTrain = mos(rand_seq(1:ceil(0.8*seqlen)));
    XTest = features(rand_seq(ceil(0.8*seqlen)+1:seqlen));
    YTest = mos(rand_seq(ceil(0.8*seqlen)+1:seqlen));

    % Define LSTM network
    numResponses = 1;
    numFeatures = size(XTrain{1},1);
    layers = [ sequenceInputLayer(numFeatures,'Normalization','zerocenter',...
                                  'NormalizationDimension','element')               
               lstmLayer(512,'OutputMode','last',...
                         'InputWeightsInitializer','narrow-normal',...
                         'RecurrentWeightsInitializer','narrow-normal') 
               fullyConnectedLayer(numResponses,...
                                   'WeightsInitializer','narrow-normal')
               huberRegressionLayer('reg')];

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
                              'L2Regularization',0.0001, ...
                              'ExecutionEnvironment','cpu', ...
                              'Shuffle','every-epoch', ...
                              'ValidationData',{XTest,YTest}, ...
                              'ValidationFrequency',100, ...
                              'Plots','none','Verbose',0);

    % Train network
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Predict the values for the test set
    YPred = predict(net,XTest,'ExecutionEnvironment','cpu')';
    new_result = [corr(YTest, YPred','type','Pearson') ...
                  corr(YTest, YPred','type','Spearman') ...
                  sqrt(mse(YTest, YPred'))];     
    results = [results; new_result];  
end

fprintf(repmat(char(8), 1, length(indicator_text)));
means = mean(results);
stds = std(results);
fprintf('Ready! Total results: \n');
fprintf('PCC %2.3f (%1.3f) SCC %2.3f (%1.3f) RMSE %0.4f (%0.4f)\n', ...
         means(1),stds(1),means(2),stds(2),means(3),stds(3));
end

% EOF