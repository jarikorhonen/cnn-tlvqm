%---------------------------------------------------------
% Example of how to use computeCNNTLVQMfeatures
% Compute features for all the videos in KoNViD-1k
%
%  Written by Jari Korhonen, Shenzhen University
%

konvid_path = '.\\KoNViD\\';
konvid_mos_file = '.\\KoNViD_mos_fr.csv';
cnn_model_file = '.\\CNN_model.mat';
feature_file = '.\\KoNViD_features.mat';

% Read subjective data from a pre-made data file
data = csvread(konvid_mos_file);
[~,idx] = sort(data(:,1));
data = data(idx,:);
file_id = data(:,1);
mos = data(:,2);
frates = data(:,3);

% Load the pretrained CNN model for spatial features
load(cnn_model_file);

% Loop through all the files to compute features
features={};
for i=1:length(file_id)

    yuv_path = sprintf('%s%d_*.yuv', konvid_path, file_id(i));
    yuv_files = dir(yuv_path);

    % Compute features for each video file
    fprintf('Computing features for sequence: %s\n',yuv_files.name);
    tic
    features{i} = computeCNNTLVQMfeatures([konvid_path yuv_files.name], [960 540], 30, netTransfer);
    toc
end
save(feature_file,'features','mos','-v7.3');
fprintf('All done!\n');
