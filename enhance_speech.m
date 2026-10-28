function [yEnhanced,infoOut] = enhance_speech(net,inputFile,outputFile,cfg)
%ENHANCE_SPEECH Enhance a noisy WAV file using the trained mask network.
%
% The signal is processed using overlapping STFT frames. The predicted
% mask is multiplied with the noisy magnitude and the original noisy phase
% is retained for ISTFT reconstruction.

if nargin < 4
    cfg = config();
end

[x,fs] = audioread(inputFile);
if size(x,2)>1
    x = mean(x,2);
end
if fs ~= cfg.fs
    x = resample(x,cfg.fs,fs);
end

[S,~,~] = stft(x,cfg.fs, ...
    Window=cfg.window, ...
    OverlapLength=cfg.winLength-cfg.hopLength, ...
    FFTLength=cfg.fftLength);

nFrames = size(S,2);
mag = abs(S);
phase = angle(S);

% Network expects 256 x 64 patches. Process in blocks of 64 frames.
enhMag = zeros(size(mag),"like",mag);

for first = 1:cfg.timeFrames:nFrames
    idx = first:min(first+cfg.timeFrames-1,nFrames);
    block = mag(:,idx);

    if size(block,2) < cfg.timeFrames
        block(:,end+1:cfg.timeFrames) = block(:,end);
    end

    block = block(1:cfg.freqBins,:);
    X = log1p(block);
    X = reshape(X,[cfg.freqBins,cfg.timeFrames,1]);
    X = (X-mean(X,"all"))/(std(X,0,"all")+1e-6);
    X = single(X);

    mask = predict(net,X);
    mask = double(squeeze(mask));

    % The network predicts 256 bins. Extend the last predicted bin to the
    % 257th FFT bin so that the mask matches the full STFT magnitude.
    maskFull = zeros(size(mag,1),cfg.timeFrames);
    maskFull(1:cfg.freqBins,:) = mask;
    maskFull(cfg.freqBins+1,:) = mask(cfg.freqBins,:);

    nUse = numel(idx);
    enhMag(:,idx) = mag(:,idx).*maskFull(:,1:nUse);
end

Senh = enhMag .* exp(1j*phase);

% ISTFT. The result is cropped back to the input duration.
yEnhanced = istft(Senh,cfg.fs, ...
    Window=cfg.window, ...
    OverlapLength=cfg.winLength-cfg.hopLength, ...
    FFTLength=cfg.fftLength);

yEnhanced = real(yEnhanced(:));

yEnhanced = yEnhanced(1:min(numel(yEnhanced),numel(x)));

if numel(yEnhanced)<numel(x)
    yEnhanced(end+1:numel(x),1)=0;
end

peak = max(abs(yEnhanced));
if peak>0.99
    yEnhanced = yEnhanced/peak*0.99;
end

if nargin >= 3 && ~isempty(outputFile)
    audiowrite(outputFile,yEnhanced,cfg.fs);
end

infoOut.inputFs = fs;
infoOut.processingFs = cfg.fs;
infoOut.frames = nFrames;
infoOut.frameStrideSeconds = cfg.hopLength/cfg.fs;
infoOut.numSamples = numel(x);
end
