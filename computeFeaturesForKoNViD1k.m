%---------------------------------------------------------
% Example of how to use computeCNNTLVQMfeatures
% Compute features for all the videos in KoNViD-1k
%

function features = computeFeaturesForKoNViD1k(konvid_path, ...
                                               konvid_feature_file, ...
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
    konvid_metadata_file = 'KoNViD_1k_attributes.csv';
    if not(isfile([konvid_path filesep konvid_metadata_file]))
        fprintf('KoNViD-1k metadata file not found!\n');
        fprintf('Make sure %s is in %s.\n', konvid_metadata_file, ...
                                            konvid_path);
        fprintf('KoNViD-1k features not extracted.\n');
        features = [];
        return;
    end
    [data,datatxt] = xlsread([konvid_path filesep konvid_metadata_file]);
    file_id = datatxt(2:end,3);
    mos = (data(:,4)-1)./4;

    % Loop through all the files to compute features
    features={};
    indicator_text = '';
    for i=1:length(file_id)

        video_path = [konvid_path filesep file_id{i}];
        fprintf(repmat(char(8), 1, length(indicator_text)));
        indicator_text = sprintf('Computing features for file %d/%d\n', ...
                                 i, length(file_id));
        fprintf(indicator_text);

        % Compute features
        features{i} = computeCNNTLVQMfeatures(video_path, netTransfer, ...
                                              cpugpu);
        if isempty(features)
            fprintf('\nFailed extracting features.\n');
            return;
        end
    end
    save(konvid_feature_file,'features','mos','-v7.3');
    fprintf('All done!\n');
end
