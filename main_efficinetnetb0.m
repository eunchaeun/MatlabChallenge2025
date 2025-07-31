%% 1. 데이터셋 로드

imds = imageDatastore(fullfile("C:\Users\Owner\Desktop\matlab_contest\data\nyu2_train", '**', '*.jpg'));
depthds = imageDatastore(fullfile("C:\Users\Owner\Desktop\matlab_contest\data\nyu2_train", '**', '*.png'));

ds = combine(imds, depthds);

% 2. 데이터 전처리 함수
inputSize = [224 224 3];
outputSize = [224 224 1];

preprocessFcn = @(data) table( ...
    {im2single(imresize(data{1}, [224 224]))}, ...
    {im2single(imresize(data{2}, outputSize(1:2)))}, ...
    'VariableNames', {'InputImage','DepthMap'});

ds = transform(ds, preprocessFcn);

% 3. 모델 설계 (Encoder-Decoder)
% EfficientNet Encoder 불러오기
encoderNet = efficientnetb0;
lgraph = layerGraph(encoderNet);

% Classification 레이어 제거
lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool', ...
    'efficientnet-b0|model|head|dense|MatMul','Softmax','classification'});

% Decoder (Upsampling) 추가
decoderLayers = [
    transposedConv2dLayer(4, 512, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv1')
    reluLayer('Name', 'relu1')

    transposedConv2dLayer(4, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv2')
    reluLayer('Name', 'relu2')

    transposedConv2dLayer(4, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv3')
    reluLayer('Name', 'relu3')

    transposedConv2dLayer(4, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv4')
    reluLayer('Name', 'relu4')

    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'upconv5')
    reluLayer('Name', 'relu5')

    % 최종 출력 (Depth map)
    convolution2dLayer(1, 1, 'Padding','same', 'Name', 'final_conv')

    regressionLayer('Name','regressionOutput')
];


% Encoder의 마지막 출력 레이어 찾기 및 Encoder와 Decoder 연결
lastLayer = 'efficientnet-b0|model|head|MulLayer';
lgraph = addLayers(lgraph, decoderLayers);
lgraph = connectLayers(lgraph, lastLayer, 'upconv1');


% 4. 학습 옵션
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'InitialLearnRate',0.001, ...
    'MiniBatchSize',16, ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'Verbose',true);

% 5. 학습 실행
net = trainNetwork(ds, lgraph, options);


%% 6. 새로운 이미지 넣어서 Depth Map 예측(테스트)
testImg = imread("C:\Users\Owner\Desktop\matlab_contest\data\nyu2_train\basement_0001a_out\1.jpg");
testImg = im2single(imresize(testImg, [227 227]));
predDepth = predict(net, testImg);

% 7. 결과 시각화
figure;
subplot(1,2,1);
imshow(testImg);
title('Input Image');

subplot(1,2,2);
imshow(rescale(predDepth)); % Depth map은 숫자이므로 rescale로 시각화
title('Predicted Depth Map');
imwrite(uint8(255 * rescale(predDepth)), 'C:\Users\Owner\Desktop\matlab_contest\predicted_depth_final.png');
