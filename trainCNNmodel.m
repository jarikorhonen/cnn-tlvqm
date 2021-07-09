% --------------------------------------------------------------
%   trainCNNmodel.m
%
%   This function trains the CNN model to be used as local
%   feature extractor.
%
%   Changes: weight initialization defined explicitely to
%   ensure backwards compatibility with other Matlab versions.
%   Tested with R2018b and R2020a.
%
%   Inputs: 
%       path:       Path to the training patches
%       model_file: Name of the file where to save the obtained
%                   CNN model
%       cpugpu:     For using CPU, set to 'cpu', and for using 
%                   GPU, set to 'gpu'
%
%   Output: dummy
%   
function res = trainCNNmodel(path, model_file, cpugpu)

    % Load probabilistic representations for quality scores
    load(['.' filesep 'LiveC_prob.mat'],'LiveC_prob');

    % Loop through all the test images to obtain source paths
    % for test images and the respective ground truth outputs
    filenames = {};
    outputs = [];
    for i=1:length(LiveC_prob(:,1))
        for j=1:36
            filenames = [filenames; sprintf('%s%s%04d_%02d.png',path,filesep,i,j)];
            outputs = [outputs; LiveC_prob(i,:)];
        end
    end
    
    % Table for training inputs (image files) and training outputs
    T = table(filenames, outputs);
    
    % Get pre-trained ResNet50 model
    net = resnet50; 

    % Modify the model for quality prediction
    newLayer1 = fullyConnectedLayer(128,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','feature_layer1','WeightsInitializer','narrow-normal');
    newLayer2 = reluLayer('Name','ReLU_128');
    newLayer3 = dropoutLayer('Name','dropout_128');
    newLayer4 = fullyConnectedLayer(5,'WeightLearnRateFactor',2,'BiasLearnRateFactor',2,'Name','fc_output','WeightsInitializer','narrow-normal');
    newLayer5 = huberRegressionLayer('huber_regression');
    lgraph = layerGraph(net);
    lgraph = replaceLayer(lgraph,'fc1000',newLayer1);
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newLayer2);
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newLayer3);
    lgraph = addLayers(lgraph, newLayer4);
    lgraph = addLayers(lgraph, newLayer5);
    lgraph = connectLayers(lgraph, 'dropout_128', 'fc_output');
    lgraph = connectLayers(lgraph, 'fc_output', 'huber_regression');
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    % Freeze layers 1-37
    layers(1:37) = freezeWeights(layers(1:37));
    lgraph = createLgraphUsingConnections(layers,connections);

    % Define training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',32, ...        
        'MaxEpochs',2, ...
        'L2Regularization',0.01, ...
        'InitialLearnRate',0.0005, ...
        'Shuffle','every-epoch', ...
        'ExecutionEnvironment',cpugpu, ...
        'Verbose',false, ...
        'Plots','training-progress');

    % Train the model
    lgraph = trainNetwork(T,'outputs',lgraph,options);
    
    % Save the final model
    netTransfer = lgraph;
    save(model_file,'netTransfer');
    res = 0;
end
