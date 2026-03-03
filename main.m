clc;
clear;
close all;

%% ============================================================
%  SPEAKER IDENTIFICATION SYSTEM
%  Methods  : k-NN and Gaussian Mixture Models (GMM)
%  Features : MFCC + Delta Coefficients
%  Dataset  : TIMIT (Selected Speakers)
%  Author   : Anantharajan Vel Murugan
% =============================================================

fprintf('==============================================\n');
fprintf('        Speaker Identification System\n');
fprintf('        Models: k-NN and GMM\n');
fprintf('==============================================\n');

%% ------------------ 1. DATASET CONFIGURATION ------------------

datasetPath = 'C:\Users\HP\Desktop\timit';   % <-- UPDATE THIS PATH

selectedSpeakers = { ...
    'dr1-fvmh0', ...
    'dr1-mcpm0', ...
    'dr2-marc0', ...
    'dr3-falk0', ...
    'dr4-maeb0'};

numSpeakers  = length(selectedSpeakers);
numTrainFiles = 6;

trainFeatures = [];
trainLabels   = [];

testFeatures  = {};
testLabels    = [];

speakerData = cell(1, numSpeakers);

fprintf('\n[Step 1] Extracting MFCC + Delta Features...\n');

%% ------------------ 2. FEATURE EXTRACTION ------------------

for s = 1:numSpeakers

    speakerFolder = fullfile(datasetPath, selectedSpeakers{s});
    audioFiles = dir(fullfile(speakerFolder, '*.wav'));

    if isempty(audioFiles)
        fprintf('[Warning] No audio found for %s\n', selectedSpeakers{s});
        continue;
    end

    fprintf('Processing Speaker %d/%d : %s\n', ...
        s, numSpeakers, selectedSpeakers{s});

    for fileIdx = 1:length(audioFiles)

        filePath = fullfile(speakerFolder, audioFiles(fileIdx).name);
        [audio, fs] = audioread(filePath);

        % ----- Preprocessing -----
        audio = audio ./ max(abs(audio));     % Normalization
        audio = filter([1 -0.97], 1, audio);  % Pre-emphasis

        % ----- MFCC + Delta -----
        [mfccCoeff, deltaCoeff, ~] = mfcc(audio, fs);
        features = [mfccCoeff, deltaCoeff];   % 26-D feature vector

        % ----- Train/Test Split -----
        if fileIdx <= numTrainFiles

            trainFeatures = [trainFeatures; features];
            trainLabels   = [trainLabels; repmat(s, size(features,1),1)];

            speakerData{s} = [speakerData{s}; features];

        else
            testFeatures{end+1} = features;
            testLabels(end+1)   = s;
        end
    end
end

testLabelsCat = categorical(testLabels, 1:numSpeakers, selectedSpeakers);

%% ------------------ 3. MODEL TRAINING ------------------

fprintf('\n[Step 2] Training Models...\n');

% ----- k-NN -----
knnModel = fitcknn( ...
    trainFeatures, ...
    categorical(trainLabels, 1:numSpeakers, selectedSpeakers), ...
    'NumNeighbors', 5, ...
    'Standardize', true);

% ----- GMM (Log-Likelihood based) -----
gmmModels = cell(1, numSpeakers);

for s = 1:numSpeakers
    gmmModels{s} = fitgmdist( ...
        speakerData{s}, ...
        8, ...
        'RegularizationValue', 0.1, ...
        'Options', statset('MaxIter', 500));
end

%% ------------------ 4. TESTING ------------------

fprintf('[Step 3] Running Predictions...\n');

knnFramePredictions = [];
gmmFramePredictions = [];
trueFrameLabels     = [];

knnFilePredictions = [];
gmmFilePredictions = [];

for i = 1:length(testFeatures)

    features = testFeatures{i};
    numFrames = size(features,1);

    trueFrameLabels = [trueFrameLabels; ...
        repmat(testLabelsCat(i), numFrames, 1)];

    % ----- k-NN Prediction -----
    knnPred = predict(knnModel, features);
    knnFramePredictions = [knnFramePredictions; knnPred];
    knnFilePredictions  = [knnFilePredictions; mode(knnPred)];

    % ----- GMM Prediction (Log-Likelihood) -----
    logLikelihood = zeros(numFrames, numSpeakers);

    for s = 1:numSpeakers
        logLikelihood(:,s) = log(pdf(gmmModels{s}, features) + eps);
    end

    [~, idx] = max(logLikelihood, [], 2);
    gmmPred = categorical(idx, 1:numSpeakers, selectedSpeakers);

    gmmFramePredictions = [gmmFramePredictions; gmmPred];
    gmmFilePredictions  = [gmmFilePredictions; mode(gmmPred)];
end

%% ------------------ 5. PERFORMANCE METRICS ------------------

knnFilePredictions = categorical(knnFilePredictions, selectedSpeakers);
gmmFilePredictions = categorical(gmmFilePredictions, selectedSpeakers);

acc_knn_file  = mean(knnFilePredictions == testLabelsCat') * 100;
acc_gmm_file  = mean(gmmFilePredictions == testLabelsCat') * 100;

acc_knn_frame = mean(knnFramePredictions == trueFrameLabels) * 100;
acc_gmm_frame = mean(gmmFramePredictions == trueFrameLabels) * 100;

fprintf('\n========= RESULTS =========\n');
fprintf('k-NN Frame Accuracy  : %.2f%%\n', acc_knn_frame);
fprintf('k-NN File Accuracy   : %.2f%%\n', acc_knn_file);
fprintf('GMM Frame Accuracy   : %.2f%%\n', acc_gmm_frame);
fprintf('GMM File Accuracy    : %.2f%%\n', acc_gmm_file);

%% ------------------ 6. VISUALIZATION ------------------

figure('Name','k-NN File Level');
confusionchart(testLabelsCat, knnFilePredictions);
title(sprintf('k-NN File Accuracy = %.2f%%', acc_knn_file));

figure('Name','k-NN Frame Level');
confusionchart(trueFrameLabels, knnFramePredictions);
title(sprintf('k-NN Frame Accuracy = %.2f%%', acc_knn_frame));

figure('Name','GMM File Level');
confusionchart(testLabelsCat, gmmFilePredictions);
title(sprintf('GMM File Accuracy = %.2f%%', acc_gmm_file));

figure('Name','GMM Frame Level');
confusionchart(trueFrameLabels, gmmFramePredictions);
title(sprintf('GMM Frame Accuracy = %.2f%%', acc_gmm_frame));

fprintf('\nCompleted: All figures generated successfully.\n');
