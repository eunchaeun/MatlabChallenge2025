% 1. 데이터셋 로드 
imFolder = "C:\Users\sec\Documents\MATLAB\nyu2_train";

imds    = imageDatastore(fullfile(imFolder,"**","*.jpg"));
depthds = imageDatastore(fullfile(imFolder,"**","*.png"));  
ds      = combine(imds, depthds);

% 2. 데이터 전처리 
inputSize  = [224 224 3];
outputSize = [224 224 1];

% depth(mm) 0~10 000 → m단위 0~10로 복원 후 single 형변환
preprocessFcn = @(d) table( ...
    { im2single(imresize(d{1},inputSize(1:2))) }, ...           
    { single(imresize(d{2},outputSize(1:2))) / 1000 }, ...         
    'VariableNames',{'InputImage','depth'});                       

ds = transform(ds, preprocessFcn);

% 3. 모델 설계 (MobileNet V2 + 5‑stage Decoder)
% 3‑1) 사전학습 Encoder 불러오기
encoderNet = mobilenetv2;                 
lgraph     = layerGraph(encoderNet);

% 3‑2) 분류 레이어 제거
lgraph = removeLayers(lgraph, { ...
           'global_average_pooling2d_1', ...
           'Logits', ...
           'Logits_softmax', ...
           'ClassificationLayer_Logits'});

% 3‑3) Decoder 정의
decoderLayers = [
    % 7×7×1280 → 14×14×512
    transposedConv2dLayer(4,512,'Stride',2,'Cropping','same','Name','up1')
    reluLayer

    % 14 → 28
    transposedConv2dLayer(4,256,'Stride',2,'Cropping','same','Name','up2')
    reluLayer

    % 28 → 56
    transposedConv2dLayer(4,128,'Stride',2,'Cropping','same','Name','up3')
    reluLayer

    % 56 → 112
    transposedConv2dLayer(4,64,'Stride',2,'Cropping','same','Name','up4')
    reluLayer

    % 112 → 224
    transposedConv2dLayer(4,32,'Stride',2,'Cropping','same','Name','up5')
    reluLayer

    convolution2dLayer(1,1,'Padding','same','Name','out')   % 채널 1
    sigmoidLayer('Name','out_sig')                          % 0~1로 제한
    regressionLayer('Name','depth')                         % 테이블 변수와 동일
];

lgraph = addLayers(lgraph, decoderLayers);

% 3‑4) Encoder‑Decoder 연결
lgraph = connectLayers(lgraph, 'out_relu', 'up1');

% 4. 학습 옵션 
options = trainingOptions('adam', ...
    'MaxEpochs',         3, ...
    'InitialLearnRate',  1e-2, ...
    'MiniBatchSize',     16, ...
    'Shuffle',           'every-epoch', ...
    'Plots',             'training-progress', ...
    'Verbose',           true);

% 5. 학습
net = trainNetwork(ds, lgraph, options);

% 6. 테스트
testImg   = im2single(imresize(imread("C:\Users\sec\Documents\MATLAB\nyu2_test\00000_colors.png"), inputSize(1:2)));
predDepth = predict(net, testImg);

% 7. 결과 시각화 
subplot(1,2,1), imshow(testImg),          title("Input Image");
subplot(1,2,2), imshow(rescale(predDepth)), title("Predicted Depth Map");
