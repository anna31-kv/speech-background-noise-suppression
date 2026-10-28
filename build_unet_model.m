function lgraph = build_unet_model(inputSize)
%BUILD_UNET_MODEL Small U-Net-style spectral mask estimator.
%
% Input:  256 x 64 x 1 noisy log-magnitude patch
% Output: 256 x 64 x 1 ideal ratio mask in [0,1]
%
% This version matches the conceptual U-Net architecture in the report:
% convolutional encoder -> bottleneck -> decoder + skip connections.

lgraph = layerGraph();

input = imageInputLayer(inputSize,Normalization="none",Name="input");
lgraph = addLayers(lgraph,input);

enc1 = [
    convolution2dLayer(3,16,Padding="same",Name="enc1_conv1")
    reluLayer(Name="enc1_relu1")
    convolution2dLayer(3,16,Padding="same",Name="enc1_conv2")
    reluLayer(Name="enc1_relu2")
    ];
lgraph = addLayers(lgraph,enc1);

pool1 = maxPooling2dLayer(2,Stride=2,Name="pool1");
lgraph = addLayers(lgraph,pool1);

enc2 = [
    convolution2dLayer(3,32,Padding="same",Name="enc2_conv1")
    reluLayer(Name="enc2_relu1")
    convolution2dLayer(3,32,Padding="same",Name="enc2_conv2")
    reluLayer(Name="enc2_relu2")
    ];
lgraph = addLayers(lgraph,enc2);

pool2 = maxPooling2dLayer(2,Stride=2,Name="pool2");
lgraph = addLayers(lgraph,pool2);

bottleneck = [
    convolution2dLayer(3,64,Padding="same",Name="bottleneck_conv1")
    reluLayer(Name="bottleneck_relu1")
    convolution2dLayer(3,64,Padding="same",Name="bottleneck_conv2")
    reluLayer(Name="bottleneck_relu2")
    ];
lgraph = addLayers(lgraph,bottleneck);

up2 = transposedConv2dLayer(2,32,Stride=2,Name="up2");
lgraph = addLayers(lgraph,up2);
cat2 = concatenationLayer(3,2,Name="cat2");
lgraph = addLayers(lgraph,cat2);

dec2 = [
    convolution2dLayer(3,32,Padding="same",Name="dec2_conv1")
    reluLayer(Name="dec2_relu1")
    convolution2dLayer(3,32,Padding="same",Name="dec2_conv2")
    reluLayer(Name="dec2_relu2")
    ];
lgraph = addLayers(lgraph,dec2);

up1 = transposedConv2dLayer(2,16,Stride=2,Name="up1");
lgraph = addLayers(lgraph,up1);
cat1 = concatenationLayer(3,2,Name="cat1");
lgraph = addLayers(lgraph,cat1);

dec1 = [
    convolution2dLayer(3,16,Padding="same",Name="dec1_conv1")
    reluLayer(Name="dec1_relu1")
    convolution2dLayer(3,16,Padding="same",Name="dec1_conv2")
    reluLayer(Name="dec1_relu2")
    convolution2dLayer(1,1,Padding="same",Name="mask_logits")
    sigmoidLayer(Name="mask")
    regressionLayer(Name="regression")
    ];
lgraph = addLayers(lgraph,dec1);

lgraph = connectLayers(lgraph,"input","enc1_conv1");
lgraph = connectLayers(lgraph,"enc1_relu2","pool1");
lgraph = connectLayers(lgraph,"pool1","enc2_conv1");
lgraph = connectLayers(lgraph,"enc2_relu2","pool2");
lgraph = connectLayers(lgraph,"pool2","bottleneck_conv1");

lgraph = connectLayers(lgraph,"bottleneck_relu2","up2");
lgraph = connectLayers(lgraph,"up2","cat2/in1");
lgraph = connectLayers(lgraph,"enc2_relu2","cat2/in2");
lgraph = connectLayers(lgraph,"cat2","dec2_conv1");

lgraph = connectLayers(lgraph,"dec2_relu2","up1");
lgraph = connectLayers(lgraph,"up1","cat1/in1");
lgraph = connectLayers(lgraph,"enc1_relu2","cat1/in2");
lgraph = connectLayers(lgraph,"cat1","dec1_conv1");
end
