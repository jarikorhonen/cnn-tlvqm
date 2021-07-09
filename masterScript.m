%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  This script runs the experiments to reproduce the results
%  for CNN-TLVQM video quality model published in ACM MM'20.
%
%  Written by Jari Korhonen, Shenzhen University
%
%  inputs: 
%          livec_path: path to the LIVE Challenge image quality  
%          database (e.g. 'd:/live_challenge')
%
%          konvid_path: path to the KoNViD-1k video quality  
%          database (e.g. 'd:/konvid1k')
%
%          livevqc_path: path to the LIVE-VQC video quality 
%          database (e.g. 'd:/livevqc')
%
%          cpugpu: defined if CPU or GPU is used for training
%          and testing the CNN model, use either 'cpu' or 'gpu'
%
%  outputs: 
%          Writes the results in file 'results.csv'
%          Plots the results shown in cross-database experiment
%          (see the original ACM Multimedia paper)
%
function out = masterScript(livec_path, ...
                            konvid_path, ...
                            livevqc_path, ...
                            cpugpu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 1: setup and testing for proper configuration
%
                        
cnn_model_file = ['.' filesep 'CNN_model.mat'];
livec_patches_path =  ['.' filesep 'livec_patches'];
konvid_feature_file = ['.' filesep 'KoNViD_features.mat'];
livevqc_feature_file = ['.' filesep 'LIVEVQC_features.mat'];
result_file = ['.' filesep 'results.csv'];

out = 0;

%-----------------------------------------------------------
% Test that KoNViD-1k database and metadata is available
%
if not(isfolder(konvid_path))
    fprintf('KoNViD-1k database not found in the given folder.\n');
    return;
end
if length(dir([konvid_path filesep '*.mp4'])) ~= 1200
    fprintf('KoNViD-1k database does not have all the video files.\n');
    return;
end
if not(isfile([konvid_path filesep 'KoNViD_1k_attributes.csv']))
    fprintf('KoNViD-1k metadata file not available.\n');
    return;
end

%-----------------------------------------------------------
% Test that LIVE-VQC database and metadata is available
%
if not(isfolder(livevqc_path))
    fprintf('LIVE-VQC database not found in the given folder.\n');
    return;
end
if length(dir([livevqc_path filesep '*.mp4'])) ~= 585
    fprintf('LIVE-VQC database does not have all the video files.\n');
    return;
end
if not(isfile([livevqc_path filesep 'LIVE_VQC.xlsx']))
    fprintf('LIVE-VQC metadata file not available.\n');
    return;
end

%-----------------------------------------------------------
% Prepare patches for training CNN feature extractor 
%
make_patches = 1;
if not(isfolder(livec_patches_path))
    mkdir(livec_patches_path);
elseif length(dir([livec_patches_path filesep '*.png'])) == 41856
    fprintf('Looks like training patches exist already.\n');
    button = input('Do you want to rewrite [y/N]?\n','s');
	if not(strcmp(button,'y')) && not(strcmp(button,'Y'))
        make_patches = 0;
    end
end

if make_patches
    if not(isfolder(livec_path))
        fprintf('LIVE Challenge dataset not found! Check the path.\n');
        return;
    else
        if length(dir([livec_path filesep 'Images' filesep '*.*'])) ~= 1166 || ...
           not(isfile([livec_path filesep 'Data' filesep 'AllMOS_release.mat'])) || ...
           not(isfile([livec_path filesep 'Data' filesep 'AllStdDev_release.mat']))
            fprintf('LIVE Challenge dataset not properly installed.\n');
            return;        
        end
    end
    fprintf('Generating patches from LIVE Challenge database...\n');
    processLiveChallenge(livec_path, livec_patches_path);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 2: train the CNN feature extractor

train_model = 1;
if isfile(cnn_model_file)
    fprintf('Looks like CNN feature extractor exists already.\n');
    button = input('Do you want to train again [y/N]?\n','s');
	if not(strcmp(button,'y')) && not(strcmp(button,'Y'))
        train_model = 0;
    end
end
if train_model
    fprintf('Training CNN feature extractor...\n');
    trainCNNmodel(livec_patches_path, cnn_model_file, cpugpu);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 3: computing features for KoNViD-1k and LIVE-VQC

%-----------------------------------------------------------
% Compute features for all the videos in KoNViD-1k
%

compute_features = 1;
if isfile(konvid_feature_file)
    fprintf('Looks like KoNViD-1k feature file exist already.\n');
    button = input('Do you want to extract again [y/N]?\n','s');
	if not(strcmp(button,'y')) && not(strcmp(button,'Y'))
        compute_features = 0;
    end
end
if compute_features  
    fprintf('Extracting features for KoNViD-1k...\n');
    computeFeaturesForKoNViD1k(konvid_path, konvid_feature_file, ...
                               cnn_model_file, cpugpu);
end

%-----------------------------------------------------------
% Compute features for all the videos in LIVE-VQC
%

compute_features = 1;
if isfile(livevqc_feature_file)
    fprintf('Looks like LIVE-VQC feature file exist already.\n');
    button = input('Do you want to extract again [y/N]?\n','s');
	if not(strcmp(button,'y')) && not(strcmp(button,'Y'))
        compute_features = 0;
    end
end
if compute_features  
    fprintf('Extracting features for LIVE-VQC...\n');
    computeFeaturesForLIVEVQC(livevqc_path, livevqc_feature_file, ...
                              cnn_model_file, cpugpu);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Phase 4: training and testing the video quality model

if isfile(result_file)
    fprintf('Looks like the result file exists already.\n');
    button = input('Overwrite or append new results [o/A]?\n','s');
	if not(strcmp(button,'o')) && not(strcmp(button,'O')) 
        f = fopen(result_file, 'a');
    else
        f = fopen(result_file, 'w');
    end
else
    f = fopen(result_file, 'w');
end


%-----------------------------------------------------------
% Test CNN on KoNViD-1k with SVR, 100 random splits
%
fprintf(f, 'Training and testing on KoNViD-1k (100 splits)\n');
if not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k feature file not found!\n');
else
    load(konvid_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    fprintf('Testing CNN (SVR) on KoNViD-1k (100 splits)\n');
    results = predictMOSwithSVR_100splits(features, mos);
    means = mean(results);
    stds = std(results);
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*4, stds(3)*4);         
    fprintf(f, 'Model, PCC (stdev), SCC (stdev), RMSE (stdev)\n');
    fprintf(f, 'CNN (SVR), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %0.3f (%0.3f)\n', means(2), stds(2), ...
                                                 means(3)*4, stds(3)*4);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on KoNViD-1k with SVR, 100 random splits
%
if not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k feature file not found!\n');
else
    load(konvid_feature_file,'features','mos');
    fprintf('Testing CNN-TLVQM (SVR) on KoNViD-1k (100 splits)\n');
    results = predictMOSwithSVR_100splits(features, mos);
    means = mean(results);
    stds = std(results);
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*4, stds(3)*4);     
    fprintf(f, 'CNN-TLVQM (SVR), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %0.3f (%0.3f)\n', means(2), stds(2), ...
                                                 means(3)*4, stds(3)*4);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on KoNViD-1k with LSTM, 100 random splits
%
if not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k feature file not found!\n');
else
    load(konvid_feature_file,'features','mos');
    fprintf('Testing CNN-TLVQM (LSTM) on KoNViD-1k (100 splits)\n');
    results = predictMOSwithLSTM_100splits(features, mos);
    means = mean(results);
    stds = std(results);   
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*4, stds(3)*4);      
    fprintf(f, 'CNN-TLVQM (LSTM), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %0.3f (%0.3f)\n', means(2), stds(2), ...
                                                 means(3)*4, stds(3)*4);
end

%-----------------------------------------------------------
% Test CNN on LIVE-VQC with SVR, 100 random splits
%
fprintf(f, '\nTraining and testing on LIVE-VQC (100 splits)\n');
if not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k feature file not found!\n');
else
    load(konvid_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    fprintf('Testing CNN (SVR) on LIVE-VQC (100 splits)\n');
    results = predictMOSwithSVR_100splits(features, mos);
    means = mean(results);
    stds = std(results);
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*100, stds(3)*100);         
    fprintf(f, 'Model, PCC (stdev), SCC (stdev), RMSE (stdev)\n');
    fprintf(f, 'CNN (SVR), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %2.2f (%2.2f)\n', means(2), stds(2), ...
                                                 means(3)*100, stds(3)*100);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on LIVE-VQC with SVR, 100 random splits
%
if not(isfile(livevqc_feature_file))
    fprintf('LIVE-VQC feature file not found!\n');
else
    load(livevqc_feature_file,'features','mos');
    fprintf('Testing CNN-TLVQM (SVR) on LIVE-VQC (100 splits)\n');
    results = predictMOSwithSVR_100splits(features, mos);
    means = mean(results);
    stds = std(results);
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*100, stds(3)*100);        
    fprintf(f, 'CNN-TLVQM (SVR), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %2.2f (%2.2f)\n', means(2), stds(2), ...
                                                 means(3)*100, stds(3)*100);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on LIVE-VQC with LSTM, 100 random splits
%
if not(isfile(livevqc_feature_file))
    fprintf('LIVE-VQC feature file not found!\n');
else
    load(livevqc_feature_file,'features','mos');
    fprintf('Testing CNN-TLVQM (LSTM) on LIVE-VQC (100 splits)\n');
    results = predictMOSwithLSTM_100splits(features, mos);
    means = mean(results);
    stds = std(results);
    fprintf('Results: PCC %2.3f (%1.3f), SCC %2.3f (%1.3f), RMSE %2.2f (%2.2f)\n', ...
            means(1), stds(1), means(2), stds(2), means(3)*100, stds(3)*100);     
    fprintf(f, 'CNN-TLVQM (LSTM), %2.3f (%1.3f), ', means(1), stds(1));
    fprintf(f, '%2.3f (%1.3f), %2.2f (%2.2f)\n', means(2), stds(2), ...
                                                 means(3)*100, stds(3)*100);
end

%-----------------------------------------------------------
% Test CNN on KoNViD-1k for training, LIVE-VQC for testing
%
fprintf(f, '\nTraining on KoNViD-1k and testing on LIVE-VQC\n');
if not(isfile(livevqc_feature_file)) || not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k or LIVE-VQC feature file(s) not found!\n');
else
    load(konvid_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    features_train = features;
    mos_train = mos;
    load(livevqc_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    features_test = features;
    mos_test = mos;
    
    fprintf('Testing CNN (SVR), training on KoNViD-1k, testing on LIVE-VQC\n');
    results = predictMOSwithSVR_CrossDB(features_train, mos_train, ...
                                         features_test, mos_test);
    fprintf('Results: ');
    fprintf('PCC %2.3f, SCC %2.3f, RMSE %2.2f\n', ...
             results(1),results(2),results(3)*100);
    fprintf(f, 'Model, PCC, SCC, RMSE\n');
    fprintf(f, 'CNN (SVR), %2.3f, %2.2f, %2.2f\n', ...
             results(1),results(2),results(3)*100);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on KoNViD-1k for training, LIVE-VQC for testing
%
if not(isfile(livevqc_feature_file)) || not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k or LIVE-VQC feature file(s) not found!\n');
else
    load(konvid_feature_file,'features','mos');
    features_train = features;
    mos_train = mos;
    load(livevqc_feature_file,'features','mos');
    features_test = features;
    mos_test = mos;
    
    fprintf('Testing CNN-TLVQM (SVR), training on KoNViD-1k, testing on LIVE-VQC\n');
    results = predictMOSwithSVR_CrossDB(features_train, mos_train, ...
                                         features_test, mos_test);
    fprintf('Results: ');
    fprintf('PCC %2.3f, SCC %2.3f, RMSE %2.2f\n', ...
             results(1),results(2),results(3)*100);
    fprintf(f, 'CNN-TLVQM (SVR), %2.3f, %2.2f, %2.2f\n', ...
             results(1),results(2),results(3)*100);
end

%-----------------------------------------------------------
% Test CNN on LIVE-VQC for training, KoNViD-1k for testing
%
fprintf(f, '\nTraining on LIVE-VQC and testing on KoNViD-1k\n');
if not(isfile(livevqc_feature_file)) || not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k or LIVE-VQC feature file(s) not found!\n');
else
    load(livevqc_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    features_train = features;
    mos_train = mos;
    load(konvid_feature_file,'features','mos');
    for i=1:length(features)
        features{i}=features{i}(:,end-127:end);
    end
    features_test = features;
    mos_test = mos;
    
    fprintf('Testing CNN (SVR), training on KoNViD-1k, testing on LIVE-VQC\n');
    results = predictMOSwithSVR_CrossDB(features_train, mos_train, ...
                                         features_test, mos_test);
    fprintf('Results: ');
    fprintf('PCC %2.3f, SCC %2.3f, RMSE %2.2f\n', ...
             results(1),results(2),results(3)*4);
         
    fprintf(f, 'Model, PCC, SCC, RMSE\n');
    fprintf(f, 'CNN (SVR), %2.3f, %2.3f, %2.3f\n', ...
             results(1),results(2),results(3)*4);
end

%-----------------------------------------------------------
% Test CNN-TLVQM on LIVE-VQC for training, KoNViD-1k for testing
%
if not(isfile(livevqc_feature_file)) || not(isfile(konvid_feature_file))
    fprintf('KoNViD-1k or LIVE-VQC feature file(s) not found!\n');
else
    load(livevqc_feature_file,'features','mos');
    features_train = features;
    mos_train = mos;
    load(konvid_feature_file,'features','mos');
    features_test = features;
    mos_test = mos;
    
    fprintf('Testing CNN-TLVQM (SVR), training on KoNViD-1k, testing on LIVE-VQC\n');
    results = predictMOSwithSVR_CrossDB(features_train, mos_train, ...
                                         features_test, mos_test);
    fprintf('Results: ');
    fprintf('PCC %2.3f, SCC %2.3f, RMSE %2.2f\n', ...
             results(1),results(2),results(3)*4);
    fprintf(f, 'CNN-TLVQM (SVR), %2.3f, %2.3f, %2.3f\n', ...
             results(1),results(2),results(3)*4);
end

fclose(f);

out = 0;
end

% End of file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
