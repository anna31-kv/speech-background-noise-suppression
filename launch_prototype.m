function launch_prototype()
%LAUNCH_PROTOTYPE Simple GUI for the trained speech enhancement prototype.
%
% Train the model first with:
%   cfg=config(); build_dataset(cfg); train_noise_suppressor(cfg);

cfg=config();
if ~isfile(cfg.modelFile)
    error("Trained model not found. Run build_dataset(cfg) and train_noise_suppressor(cfg) first.");
end
load(cfg.modelFile,"net");

fig=uifigure("Name","Deep Learning Speech Noise Suppression","Position",[100 100 900 600]);

ax=uiaxes(fig,"Position",[50 220 800 330]);
title(ax,"Waveform");
xlabel(ax,"Time (s)");
ylabel(ax,"Amplitude");
grid(ax,"on");

status=uilabel(fig,"Position",[50 175 800 25],"Text","Ready.");

uibutton(fig,"push","Position",[50 115 180 40],"Text","Load Noisy WAV", ...
    "ButtonPushedFcn",@loadAudio);

uibutton(fig,"push","Position",[250 115 180 40],"Text","Enhance Speech", ...
    "ButtonPushedFcn",@enhanceAudio);

uibutton(fig,"push","Position",[450 115 180 40],"Text","Play Enhanced", ...
    "ButtonPushedFcn",@playEnhanced);

uibutton(fig,"push","Position",[650 115 180 40],"Text","Save Enhanced", ...
    "ButtonPushedFcn",@saveEnhanced);

noisy=[];
enhanced=[];
fs=cfg.fs;
inputPath="";

    function loadAudio(~,~)
        [f,p]=uigetfile("*.wav","Select noisy speech");
        if isequal(f,0), return; end
        inputPath=fullfile(p,f);
        [noisy,fs]=audioread(inputPath);
        if size(noisy,2)>1, noisy=mean(noisy,2); end
        t=(0:numel(noisy)-1)/fs;
        plot(ax,t,noisy);
        title(ax,"Noisy Speech");
        status.Text="Loaded: "+string(f);
    end

    function enhanceAudio(~,~)
        if isempty(noisy)
            uialert(fig,"Load a noisy WAV file first.","No input");
            return;
        end
        temp=fullfile(cfg.enhancedDir,"gui_enhanced.wav");
        [enhanced,info]=enhance_speech(net,inputPath,temp,cfg);
        t=(0:numel(enhanced)-1)/cfg.fs;
        plot(ax,t,enhanced);
        title(ax,"Enhanced Speech");
        status.Text=sprintf("Enhanced. Inference frames: %d; frame stride: %.2f ms", ...
            info.frames,info.frameStrideSeconds*1000);
    end

    function playEnhanced(~,~)
        if isempty(enhanced)
            uialert(fig,"Enhance the speech first.","No enhanced audio");
            return;
        end
        soundsc(enhanced,cfg.fs);
        status.Text="Playing enhanced speech...";
    end

    function saveEnhanced(~,~)
        if isempty(enhanced)
            uialert(fig,"Enhance the speech first.","No enhanced audio");
            return;
        end
        [f,p]=uiputfile("enhanced_speech.wav","Save enhanced speech");
        if isequal(f,0), return; end
        audiowrite(fullfile(p,f),enhanced,cfg.fs);
        status.Text="Saved: "+string(fullfile(p,f));
    end
end
