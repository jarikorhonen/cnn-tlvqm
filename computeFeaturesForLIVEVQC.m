%---------------------------------------------------------
% Example of how to use computeCNNTLVQMfeatures
% Compute features for all the videos in KoNViD-1k
%

function features = computeFeaturesForLIVEVQC(livevqc_path, ...
                                              livevqc_feature_file, ...
                                              cnnfile, ...
                                              cpugpu)
    
    % Load CNN feature extractor 
    if not(isfile(cnnfile))
        fprintf('CNN feature extractor not found!\n');
        features = [];
        return;
    end
    load(cnnfile,'netTransfer');                                       
                                           
    % Read metadata for KoNViD-1k
    livevqc_metadata_file = 'data.mat';
    if not(isfile([livevqc_path filesep livevqc_metadata_file]))
        fprintf('LIVE-VQC metadata file not found!\n');
        fprintf('Make sure %s is in %s.\n', livevqc_metadata_file, ...
                                            livevqc_path);
        fprintf('LIVE-VQC features not extracted.\n');
        features = [];
        return;
    end
    load([livevqc_path filesep livevqc_metadata_file]);
    mos = mos./100;

    % Loop through all the files to compute features
    features={};
    indicator_text = '';
    for i=1:length(video_list)

        video_path = [livevqc_path filesep video_list{i}];
        fprintf(repmat(char(8), 1, length(indicator_text)));
        indicator_text = sprintf('Computing features for file %d/%d\n', ...
                                 i, length(video_list));
        fprintf(indicator_text);

        % Compute features
        features{i} = computeCNNTLVQMfeatures(video_path, netTransfer, ...
                                              cpugpu);
        if isempty(features)
            fprintf('\nFailed extracting features.\n');
            return;
        end
    end
    save(livevqc_feature_file,'features','mos','-v7.3');
    fprintf('All done!\n');
end
