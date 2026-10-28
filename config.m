function cfg = config()
%CONFIG Central configuration for the speech noise suppression prototype.

cfg.fs = 16000;

% STFT parameters
cfg.fftLength = 512;
cfg.winLength = 512;
cfg.hopLength = 256;
cfg.window = hann(cfg.winLength,"periodic");

% Fixed model patch size.
% We keep 256 frequency bins and 64 time frames.
cfg.freqBins = 256;
cfg.timeFrames = 64;

% Each patch is approximately 1.04 seconds at 16 kHz.
cfg.segmentSamples = (cfg.timeFrames-1)*cfg.hopLength + cfg.winLength;

% Dataset sizes. Increase for the final experiment.
cfg.numTrain = 1000;
cfg.numVal   = 100;
cfg.numTest  = 100;

% Random SNR range for supervised mixtures.
cfg.snrRange = [-5 10];

% Model
cfg.useUNet = true;  % true = U-Net-style model; false = compact CNN

% Training
cfg.miniBatchSize = 16;
cfg.maxEpochs = 20;
cfg.initialLearnRate = 1e-3;
cfg.validationFrequency = 50;

% Paths
root = fileparts(mfilename("fullpath"));
cfg.dataRoot = fullfile(root,"data");
cfg.cleanDir = fullfile(cfg.dataRoot,"clean");
cfg.noiseDir = fullfile(cfg.dataRoot,"noise");
cfg.outputDir = fullfile(root,"outputs");
cfg.modelFile = fullfile(cfg.outputDir,"speechNoiseSuppressor.mat");
cfg.datasetFile = fullfile(cfg.outputDir,"dataset.mat");
cfg.enhancedDir = fullfile(cfg.outputDir,"enhanced");

if ~exist(cfg.outputDir,"dir"), mkdir(cfg.outputDir); end
if ~exist(cfg.enhancedDir,"dir"), mkdir(cfg.enhancedDir); end
end
