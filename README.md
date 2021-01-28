# CNN-TLVQM

CNN-TLVQM is an improved version of [Two-Level Video Quality Model (TLVQM)](https://github.com/jarikorhonen/nr-vqa-consumervideo), where the spatial high complexity (HC) features are replaced by a feature extractor based on convolutional neural network (CNN). Description of the model and the validation study has been published in ACM Multimedia 2020 (link to the paper [here](https://dl.acm.org/doi/10.1145/3394171.3413845)).

For reproducing the model and the results for KoNViD-1k and LIVE-VQC datasets, you need to first download the required third-party databases: 

LIVE Challenge image quality database from: http://live.ece.utexas.edu/research/ChallengeDB/

KoNViD-1k video quality database from: http://database.mmsp-kn.de/konvid-1k-database.html

LIVE-VQC video quality database from: https://live.ece.utexas.edu/research/LIVEVQC/index.html

The model is implemented in Matlab (we have used version R2018b), including Image Processing Toolbox and Deep Learning Toolbox. In addition, for training the CNN model from scratch, pre-trained ResNet50 is needed, and for using GPU, Parallel Processing Toolbox is needed. These can be downloaded and installed using Matlab standard add-on tool.

For reproducing the results in ACM MM paper, you can use the script `masterScript(livec_path, konvid_path, livevqc_path, cpugpu)` as in the example below:

```
>> masterScript('c:\\live_challenge', 'c:\\konvid', 'c:\\live-vqc', 'gpu');
```

In the example above, it is assumed that the LIVE Challenge, KoNViD-1k, and LIVE-VQC have been installed in directories _c:\live_challenge_, _c:\konvid_, and _c:\live-vqc_, respectively. The fourth parameter can be set to either `'cpu'` or `'gpu'`, and defines whether CPU or GPU is used for training and testing the CNN model. The script writes the results in CSV file _results.csv_ in the current directory.

The script goes automatically through the following steps in the workflow: 

### 1) Creating training images for CNN

Training images (patches) are created from LIVE in the Wild Image Quality Challenge Database (LIVE Challenge) to train the 2D-CNN model. 

In Matlab, the training data (224x224 patches) can be created by using:
```
>> processLiveChallenge(path, out_path);
```
where _path_ is the path to the LIVE Challenge database (e.g. `'c:\\live_challenge'`) and _out_path_ is the path to the produced training patches (e.g. `'.\\training_patches'`). The script will produce training images and their respective probabilistic quality scores and store them in Matlab data file _LiveC_prob.mat_ in the current path.


### 2) Training the CNN model by using the created training images 

For training the CNN feature extractor, use:
```
>> trainCNNmodel(path, cnnfile, cpugpu);
```
where _path_ is the path to the training patches (e.g. `'.\\training_patches'`), _cnnfile_ is the file where the model will be saved (e.g. `'CNN_model.mat'`), and _cpugpu_ is either `'cpu'` or `'gpu'`.

You can also download pre-trained model for Matlab [here](https://mega.nz/file/Tdxi1IAQ#_G6y6UXcOdjPsWaVhVULPcqwMNmh0YW26Jhg-pcC6aY).

### 3) Extracting the video features from KoNViD-1k and LIVE-VQC databases 

For extracting KoNViD-1k features, use:
```
>>  computeFeaturesForKoNViD1k(konvid_path, konvid_feature_file, cnnfile, cpugpu);
```
where _konvid_path_ is the path to KoNViD-1k dataset (e.g. `'c:\\konvid'`), _konvid_feature_file_ defines the Matlab data file where the features will be saved (e.g. `'.\\konvid_features.mat'`), _cnnfile_ is the file for the CNN model (e.g. `'CNN_model.mat'`), and _cpugpu_ is either `'cpu'` or `'gpu'`. Note that KoNViD-1k metadata file _KoNViD_1k_attributes.csv_ must be in the database folder.

For extracting LIVE-VQC features, use:
```
>>  computeFeaturesForLIVEVQC(livevqc_path, livevqc_feature_file, cnnfile, cpugpu);
```
where _livevqc_path_ is the path to LIVE-VQC dataset (e.g. 'c:\\live-vqc'), _livevqc_feature_file_ defines the Matlab data file where the features will be saved (e.g. `'.\\livevqc_features.mat'`), _cnnfile_ is the file for the CNN model (e.g. `'CNN_model.mat'`), and _cpugpu_ is either `'cpu'` or `'gpu'`.

### 4) Training and testing the regression models 

There are two scripts for training and testing the regression model with 100 random splits by using for SVR and LSTM, respectively. They can be used as:
```
>> results = predictMOSwithSVR_100splits(features, mos);
>> results = predictMOSwithLSTM_100splits(features, mos);
```
where _features_ contains the features for each video, as computed in the previous step, and _mos_ contains the MOS values for each video, respectively. As output, the function returns a matrix of size (100,3), where the columns represent the Pearson Linear Correlation Coefficient (PCC), Spearman Rank-order Correlation Coefficient (SCC) and Root Mean Squared Error (RMSE) for each test split. Note that RMSE has been computed for MOS values normalized to interval 0-1.  

For cross-database test, use:
```
>> results = predictMOSwithSVR_CrossDB(features_train, mos_train, features_test, mos_test);
```
where _features_train_ contains the features for the training dataset, _mos_train_ contains the respective MOS values, and _fetures_test_ and _mos_test_ contain the features and MOS values for the testing dataset, respectively. As a result, the function returns a vector with PCC, SCC, and RMSE. Usage example can be found in `masterScript`.
