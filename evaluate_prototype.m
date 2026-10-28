function results = evaluate_prototype(cfg)
%EVALUATE_PROTOTYPE Evaluate the trained speech enhancement model.
%
% Metrics:
%   - Input SNR
%   - Output SNR
%   - SNR improvement
%   - Input STOI
%   - Enhanced STOI

if nargin < 1
    cfg = config();
end

% Load trained model
if ~isfile(cfg.modelFile)
    error("Trained model not found. Run train_noise_suppressor(cfg) first.");
end

load(cfg.modelFile,"net");

% Load test data
if ~isfile(cfg.datasetFile)
    error("Dataset not found. Run build_dataset(cfg) first.");
end

load(cfg.datasetFile,"testClean","testNoise","testMixture");

N = numel(testMixture);

inSNR = zeros(N,1);
outSNR = zeros(N,1);

inputSTOI = zeros(N,1);
enhancedSTOI = zeros(N,1);

for i = 1:N

    clean = testClean{i};
    noise = testNoise{i};
    mix = testMixture{i};

    % Save noisy input temporarily
    tempIn = fullfile(cfg.enhancedDir, ...
        sprintf("test_%03d_input.wav",i));

    tempOut = fullfile(cfg.enhancedDir, ...
        sprintf("test_%03d_enhanced.wav",i));

    audiowrite(tempIn,mix,cfg.fs);

    % Enhance speech
    [enh,~] = enhance_speech(net,tempIn,tempOut,cfg);

    % Make all signals the same length
    L = min([numel(clean),numel(noise),numel(enh)]);

    clean = clean(1:L);
    noise = noise(1:L);
    mix = mix(1:L);
    enh = enh(1:L);

    % ------------------------------------------------
    % SNR
    % ------------------------------------------------

    inSNR(i) = 10*log10( ...
        sum(clean.^2) / ...
        (sum(noise.^2) + eps));

    residual = enh - clean;

    outSNR(i) = 10*log10( ...
        sum(clean.^2) / ...
        (sum(residual.^2) + eps));

    % ------------------------------------------------
    % STOI
    % ------------------------------------------------

    inputSTOI(i) = stoi(clean,mix,cfg.fs);

    enhancedSTOI(i) = stoi(clean,enh,cfg.fs);

    % ------------------------------------------------
    % Plot first example
    % ------------------------------------------------

    if i == 1
        plot_example(clean,mix,enh,cfg.fs);
    end

end

% ------------------------------------------------
% Calculate average results
% ------------------------------------------------

results.meanInputSNR = mean(inSNR);

results.meanOutputSNR = mean(outSNR);

results.meanSNRImprovement = ...
    mean(outSNR - inSNR);

results.meanInputSTOI = mean(inputSTOI);

results.meanEnhancedSTOI = mean(enhancedSTOI);

results.meanSTOIImprovement = ...
    mean(enhancedSTOI - inputSTOI);

% ------------------------------------------------
% Display results
% ------------------------------------------------

fprintf("\n");
fprintf("=============================================\n");
fprintf("     Speech Noise Suppression Evaluation\n");
fprintf("=============================================\n");

fprintf("Mean input SNR       : %.2f dB\n", ...
    results.meanInputSNR);

fprintf("Mean output SNR      : %.2f dB\n", ...
    results.meanOutputSNR);

fprintf("Mean SNR improvement : %.2f dB\n", ...
    results.meanSNRImprovement);

fprintf("\n");

fprintf("Mean input STOI      : %.4f\n", ...
    results.meanInputSTOI);

fprintf("Mean enhanced STOI   : %.4f\n", ...
    results.meanEnhancedSTOI);

fprintf("Mean STOI improvement: %.4f\n", ...
    results.meanSTOIImprovement);

fprintf("=============================================\n");

% Save results

save(fullfile(cfg.outputDir,"results.mat"), ...
    "results", ...
    "inSNR", ...
    "outSNR", ...
    "inputSTOI", ...
    "enhancedSTOI");

end


function plot_example(clean,noisy,enh,fs)

t = (0:numel(clean)-1)/fs;

figure( ...
    "Name","Speech Noise Suppression Prototype", ...
    "Color","w");

plot(t,noisy);
hold on;

plot(t,enh);

plot(t,clean);

grid on;

xlabel("Time (s)");
ylabel("Amplitude");

legend( ...
    "Noisy", ...
    "Enhanced", ...
    "Clean", ...
    Location="best");

title("Before vs After Speech Enhancement");

end