# CNN-TLVQM

![CNN-TLVQM diagram](https://github.com/jarikorhonen/cnn-tlvqm/blob/master/cnn-tlvqm_small.png)

The instructions for reproducing the model and the results for KoNViD-1k dataset using CNN-TLVQM video quality model are given here.

The model is implemented in Matlab (we have used version R2018b), including Image Processing Toolbox and Deep Learning Toolbox.

There are four steps in the workflow for using the model: 

### 1) Creating training images for CNN

Training images (patches) are created from LIVE in the Wild Image Quality Challenge Database (LIVE Challenge) to train the 2D-CNN model. LIVE Challenge database can be obtained from: http://live.ece.utexas.edu/research/ChallengeDB/.

In Matlab, the training data (224x224 patches) can be created by using:
```
>> processLiveChallenge(path, out_path);
```
where path is the path to the LIVE Challenge database (e.g. 'c:\\LiveChallenge') and out_path is the path to the produced training images (e.g. 'c:\\Training_Images'). The script will produce training images and their respective probabilistic quality scores, stored in Matlab data file 'LiveC_prob.mat' in the folder out_path.


### 2) Training the CNN model by using the created training images 

In Matlab, use:
```
>> trainCNNmodel(path, model_file);
```
where path is the path to the training images (e.g. 'c:\\Training_Images'), and model_file is the file where the model will be saved (e.g. c:\\CNN_model.mat').

Note that in Matlab 2018b, for freezeWeights and createLgraphUsingConnections, you may need to first add matlabroot\examples\nnet\ in the path, by using:
```  
>> addpath([matlabroot,'\\examples\\nnet\\main']);
```

### 3) Extracting the sequences of features from KoNViD-1k video sequences 

First, you need to obtain KoNVid-1k from: http://database.mmsp-kn.de/konvid-1k-database.html. Note that the video files in the database are distributed as compressed MP4 files, and you need to decompress them to YUV4:2:0 format (\*.yuv files) before extracting the features. You can use e.g. ffmpeg (http://ffmpeg.org) for decoding.

Then, you can use Matlab script:
```
>>  computeFeaturesForKonvid1k.m 
```
Note that you need to change the file names and paths in the script as follows:
```
konvid_path: path to the KoNViD-1k database (e.g. 'c:\\KoNViD-1k').

konvid_mos_file: the file with pre-processed KoNViD-1k MOS values and frame rates, included in this zip file (by default, '.\\konvid_mos_fr.csv').

cnn_model_file: the file for the CNN model trained in step 2 (e.g. 'c:\\CNN_model.mat').

feature_file: path to the folder where the resulting feature file is saved (e.g. 'c:\\KoNViD-1k\\KoNViD_features.mat'). The script will save the features in Matlab data file 'KONVID_features.mat'.
```

### 4) Training and testing the regression models 

There are two scripts for this purpose, for SVR and LSTM, respectively. They can be used as:
```
>> predictMOSwithLSTM.m
>> predictMOSwithSVR.m
```
You need to change the following filenames in the scripts:

feature_file: the file where the features are saved (e.g. 'c:\\KoNViD-1k\\KONVID_features.mat').
result_file: the file where to save the results (e.g. 'c:\\KoNViD-1k\\KONVID_results_LSTM.csv' or 'c:\\KoNViD-1k\\KONVID_results_SVR.csv').
