function layers = build_model(inputSize)
%BUILD_MODEL Compact CNN spectral-mask estimator.
%
% Input : normalized noisy magnitude spectrogram
% Output: IRM in [0,1] with the same spatial dimensions.

layers = [
    imageInputLayer(inputSize,Normalization="none",Name="input")

    convolution2dLayer(3,16,Padding="same",Name="conv1")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")

    convolution2dLayer(3,32,Padding="same",Name="conv2")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")

    convolution2dLayer(3,32,Padding="same",Name="conv3")
    batchNormalizationLayer(Name="bn3")
    reluLayer(Name="relu3")

    convolution2dLayer(3,16,Padding="same",Name="conv4")
    batchNormalizationLayer(Name="bn4")
    reluLayer(Name="relu4")

    convolution2dLayer(1,1,Padding="same",Name="mask_logits")
    sigmoidLayer(Name="mask")
    regressionLayer(Name="regression")
    ];
end
