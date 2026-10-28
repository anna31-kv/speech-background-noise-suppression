function train_noise_suppressor(cfg)
%TRAIN_NOISE_SUPPRESSOR Train the compact CNN mask estimator.

if nargin < 1
    cfg = config();
end

if ~isfile(cfg.datasetFile)
    error("Dataset not found. Run build_dataset(cfg) first.");
end

load(cfg.datasetFile,"XTrain","YTrain","XVal","YVal");

inputSize = [cfg.freqBins cfg.timeFrames 1];
if cfg.useUNet
    lgraph = build_unet_model(inputSize);
    layers = lgraph;
else
    layers = build_model(inputSize);
end

options = trainingOptions("adam", ...
    InitialLearnRate=cfg.initialLearnRate, ...
    MaxEpochs=cfg.maxEpochs, ...
    MiniBatchSize=cfg.miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={XVal,YVal}, ...
    ValidationFrequency=cfg.validationFrequency, ...
    Verbose=true, ...
    Plots="training-progress", ...
    ExecutionEnvironment="auto");

fprintf("Training network...\n");
net = trainNetwork(XTrain,YTrain,layers,options);

save(cfg.modelFile,"net","cfg","-v7.3");
fprintf("Saved trained network to %s\n",cfg.modelFile);
end
