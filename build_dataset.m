function build_dataset(cfg)
%BUILD_DATASET Create fixed-size supervised spectrogram patches.
%
% The target is the ideal ratio mask (IRM). Clean and noise recordings are
% mixed at a random SNR.

if nargin < 1
    cfg = config();
end

cleanFiles = dir(fullfile(cfg.cleanDir,"**","*.wav"));
noiseFiles = dir(fullfile(cfg.noiseDir,"**","*.wav"));

if isempty(cleanFiles)
    error("No clean WAV files found in %s",cfg.cleanDir);
end
if isempty(noiseFiles)
    error("No noise WAV files found in %s",cfg.noiseDir);
end

fprintf("Found %d clean files and %d noise files.\n",numel(cleanFiles),numel(noiseFiles));

XTrain = zeros(cfg.freqBins,cfg.timeFrames,1,cfg.numTrain,"single");
YTrain = zeros(size(XTrain),"single");
XVal   = zeros(cfg.freqBins,cfg.timeFrames,1,cfg.numVal,"single");
YVal   = zeros(size(XVal),"single");
XTest  = zeros(cfg.freqBins,cfg.timeFrames,1,cfg.numTest,"single");
YTest  = zeros(size(XTest),"single");

testClean = cell(cfg.numTest,1);
testNoise = cell(cfg.numTest,1);
testMixture = cell(cfg.numTest,1);

rng(42);

for i = 1:cfg.numTrain
    [XTrain(:,:,:,i),YTrain(:,:,:,i)] = makeExample(cleanFiles,noiseFiles,cfg);
    if mod(i,50)==0, fprintf("Train %d/%d\n",i,cfg.numTrain); end
end

for i = 1:cfg.numVal
    [XVal(:,:,:,i),YVal(:,:,:,i)] = makeExample(cleanFiles,noiseFiles,cfg);
end

for i = 1:cfg.numTest
    [XTest(:,:,:,i),YTest(:,:,:,i),c,n,m] = makeExample(cleanFiles,noiseFiles,cfg);
    testClean{i}=c;
    testNoise{i}=n;
    testMixture{i}=m;
end

save(cfg.datasetFile,"XTrain","YTrain","XVal","YVal","XTest","YTest", ...
    "testClean","testNoise","testMixture","-v7.3");

fprintf("Saved dataset to %s\n",cfg.datasetFile);
end

function [X,Y,clean,noise,mix] = makeExample(cleanFiles,noiseFiles,cfg)

ci = randi(numel(cleanFiles));
ni = randi(numel(noiseFiles));

clean = audioread(fullfile(cleanFiles(ci).folder,cleanFiles(ci).name));
noise = audioread(fullfile(noiseFiles(ni).folder,noiseFiles(ni).name));

clean = mean(clean,2);
noise = mean(noise,2);

clean = resampleIfNeeded(clean,cfg.fs,fullfile(cleanFiles(ci).folder,cleanFiles(ci).name));
noise = resampleIfNeeded(noise,cfg.fs,fullfile(noiseFiles(ni).folder,noiseFiles(ni).name));

clean = randomSegment(clean,cfg.segmentSamples);
noise = randomSegment(noise,cfg.segmentSamples);

% Remove DC.
clean = clean - mean(clean);
noise = noise - mean(noise);

% RMS-normalize noise and scale it to the requested SNR.
crms = sqrt(mean(clean.^2)+eps);
nrms = sqrt(mean(noise.^2)+eps);

snrDb = cfg.snrRange(1) + rand*(cfg.snrRange(2)-cfg.snrRange(1));
noise = noise/nrms * crms / (10^(snrDb/20));

mix = clean + noise;

% Prevent clipping in stored waveform.
peak = max(abs(mix));
if peak > 0.99
    mix = mix/peak*0.99;
    clean = clean/peak*0.99;
    noise = noise/peak*0.99;
end

[Sx,Sc] = localSTFT(mix,cfg);
[Sclean,~] = localSTFT(clean,cfg);
[Snoise,~] = localSTFT(noise,cfg);

magX = abs(Sx);
magC = abs(Sclean);
magN = abs(Snoise);

% Ideal Ratio Mask.
Yfull = magC ./ (magC + magN + 1e-8);
Xfull = log1p(magX);

X = fitPatch(Xfull,cfg);
Y = fitPatch(Yfull,cfg);

% Normalize each input patch.
mu = mean(X,"all");
sd = std(X,0,"all") + 1e-6;
X = (X-mu)/sd;

X = single(X);
Y = single(Y);
end

function y = randomSegment(x,N)
if numel(x) < N
    x = [x; zeros(N-numel(x),1)];
end
maxStart = numel(x)-N+1;
start = randi(maxStart);
y = x(start:start+N-1);
end

function y = resampleIfNeeded(x,fs,targetFile)
info = audioinfo(targetFile);
if info.SampleRate ~= fs
    y = resample(x,fs,info.SampleRate);
else
    y = x;
end
end

function [S,phase] = localSTFT(x,cfg)
[S,~,~] = stft(x,cfg.fs, ...
    Window=cfg.window, ...
    OverlapLength=cfg.winLength-cfg.hopLength, ...
    FFTLength=cfg.fftLength);
phase = angle(S);
end

function A = fitPatch(A,cfg)
if size(A,1) < cfg.freqBins
    A(end+1:cfg.freqBins,:) = 0;
end
A = A(1:cfg.freqBins,:);

if size(A,2) < cfg.timeFrames
    A(:,end+1:cfg.timeFrames) = A(:,end);
end
A = A(:,1:cfg.timeFrames);
A = reshape(A,[cfg.freqBins,cfg.timeFrames,1]);
end
