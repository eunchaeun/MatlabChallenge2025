%%
net = squeezenet;
lgraph = layerGraph(net);

%기존 출력 부분 제거 (분류 레이어)
lgraph = removeLayers(lgraph, {'conv10', 'relu_conv10', 'pool10', 'prob', 'ClassificationLayer_predictions'});

%업샘플링 계층 추가 (stride=2로 점차 해상도 높임)
layers = [
    transposedConv2dLayer(3, 256, 'Stride', 2, 'Cropping', 'same', 'Name', 'up1')
    reluLayer('Name','relu1')

    transposedConv2dLayer(3, 128, 'Stride', 2, 'Cropping', 'same', 'Name', 'up2')
    reluLayer('Name','relu2')

    transposedConv2dLayer(3, 64, 'Stride', 2, 'Cropping', 'same', 'Name', 'up3')
    reluLayer('Name','relu3')

    transposedConv2dLayer(4, 32, 'Stride', 2, 'Cropping', 'same', 'Name', 'up4')
    reluLayer('Name','relu4')

    convolution2dLayer(3,1,'Padding','same','Name','finalConv')
    regressionLayer('Name','regressionOutput')
];

lgraph = addLayers(lgraph, layers);

%기존 feature layer와 업샘플링 연결 (drop9 → up1)
lgraph = connectLayers(lgraph, 'drop9', 'up1');

% --- 데이터셋 불러오기 ---
imds = imageDatastore(fullfile("C:\Users\Owner\Desktop\matlab_contest\data\nyu2_train", '**', '*.jpg'));
depthds = imageDatastore(fullfile("C:\Users\Owner\Desktop\matlab_contest\data\nyu2_train", '**', '*.png'));
ds = combine(imds, depthds);

% --- 전처리 함수 정의 ---
inputSize = [227 227];
outputSize = [224 224];

function dataOut = preprocessData(data, inputSize, outputSize)
    img = im2single(imresize(data{1}, inputSize));
    depth = im2single(imresize(data{2}, outputSize));

    if ndims(depth) == 2
        depth = reshape(depth, [size(depth,1), size(depth,2), 1]);
    end

    dataOut = {img, depth};
end

% --- 데이터셋 변환 적용 ---
dsTrain = transform(ds, @(data) preprocessData(data, inputSize, outputSize));

% --- 학습 옵션 ---
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% --- 학습 실행 ---
trainedNet = trainNetwork(dsTrain, lgraph, options);


%%
% 6. 테스트 --------------------------------------------------------------
testImg = imread("C:\Users\Owner\Desktop\matlab_contest\finalpeople.png");
testImg = im2single(imresize(testImg, [227 227]));
predDepth = predict(trainedNet, testImg);

% 7. 결과 시각화 ---------------------------------------------------------
figure;
subplot(1,2,1), imshow(testImg),          title("Input Image");
subplot(1,2,2), imshow(rescale(predDepth)), title("Predicted Depth Map");

imwrite(uint8(255 * rescale(predDepth)), 'C:\Users\Owner\Desktop\matlab_contest\predicted_depth_final.png');


