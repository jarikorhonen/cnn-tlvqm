% --------------------------------------------------------------
%   processLiveChallenge.m
%
%   Written by Jari Korhonen, Shenzhen University
%
%   This function processes LIVE Image Quality Challenge
%   database to produce the training images and obtain the
%   approximated probabilistic representations for the quality
%   scores.
%
%   Usage: processLiveChallenge(path)
%   Inputs: 
%       path: string with the path to LIVE Challenge database
%   Outuput: dummy
%
%   Note that you need to change the paths in this file to the 
%   actual folder where the database is located.
%   

function res = processLiveChallenge(path, out_path)

    % First, load the MOS scores and stadard deviations
    % Make sure to change the paths to point to the database
    load([path '\\data\\allmos_release.mat']);
    load([path '\\data\\allstddev_release.mat']);

    % Use truncated Gaussian distribution to approximate 5-point
    % probabilistic representation of the quality scores
    bins = [];
    for i=8:length(AllMOS_release)
        b1 = truncGaussianCDF(20,AllMOS_release(i), AllStdDev_release(i),0,100);
        b2 = truncGaussianCDF(40,AllMOS_release(i), AllStdDev_release(i),0,100)-b1;
        b3 = truncGaussianCDF(60,AllMOS_release(i), AllStdDev_release(i),0,100)-b2-b1;
        b4 = truncGaussianCDF(80,AllMOS_release(i), AllStdDev_release(i),0,100)-b3-b2-b1;
        b5 = 1-b4-b3-b2-b1;
        bins = [bins; b1 b2 b3 b4 b5];
    end

    % Save the results
    LiveC_prob = bins;
    save([out_path '\\LiveC_prob.mat'],'LiveC_prob');

    % Load image names in Live Challenge database
    load([path '\\data\\allImages_release.mat']);
    patch_size = [224 224];

    % Loop through all the test images (skip the first seven for training
    for im_no=8:length(AllImages_release)

        imfile = sprintf('%s\\Images\\%s', path, AllImages_release{im_no});
        
        % Initialize variables
        im = imread(imfile);
        [height,width,~] = size(im);
        x_numb = ceil(width/patch_size(1));
        y_numb = ceil(height/patch_size(2));
        x_step = 1;
        y_step = 1;
        if x_numb>1 && y_numb>1
            x_step = floor((width-patch_size(1))/(x_numb-1));
            y_step = floor((height-patch_size(2))/(y_numb-1));
        end
        im_patches = [];
        num_patch = 1;

        % Extract patches from the image
        for i=1:x_step:width-patch_size(1)+1
            for j=1:y_step:height-patch_size(2)+1
                y_range = j:j+patch_size(2)-1;
                x_range = i:i+patch_size(1)-1;
                im_patch = im(y_range, x_range,:);
                
                % Make four rotated versions of each patch
                for q=1:4
                    filename = sprintf('%s\\%04d_%02d.png', ...
                                       out_path, im_no-7, num_patch);               
                                   
                    imwrite(im_patch,filename); 
                    im_patch = imrotate(im_patch,90);
                    num_patch = num_patch + 1;
                end
            end
        end 
    end
    res = 0;   
end

% Truncated Gaussian cumulative distribution function
function X = truncGaussianCDF(x,my,sigma,a,b)

    if x<=a 
        X=0;
    elseif x>=b
        X=1;
    else
        X = (Th(my,sigma,x)-Th(my,sigma,a))/ ...
            (Th(my,sigma,b)-Th(my,sigma,a));
    end
end

% Theta function for computing truncated Gaussian cdf
function X = Th(my,sigma,x)
    X = (1+erf((x-my)/(sigma*sqrt(2))))/2;
end