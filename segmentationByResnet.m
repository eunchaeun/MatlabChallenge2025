
% 1. 경로 설정 및 클래스 정의
imageFolder = "C:\Users\Owner\Desktop\matlab_contest\person_data\images";
labelFolder = "C:\Users\Owner\Desktop\matlab_contest\person_data\masks";

classNames = ["background", "person"];
labelIDs = [0, 255];  % 마스크 픽셀 값 (배경=0, 사람=255)

imageSize = [256 256 3];
numClasses = numel(classNames);

% 2. 데이터스토어 생성 및 결합
imds = imageDatastore(imageFolder);
pxds = pixelLabelDatastore(labelFolder, classNames, labelIDs);
trainingData = combine(imds, pxds);

function dataOut = augmentData(data, imageSize)
    inputImage = data{1};
    labelImage = data{2};

    inputImage = imresize(inputImage, imageSize(1:2));
    labelImage = imresize(labelImage, imageSize(1:2), 'nearest');
    if isnumeric(labelImage) && size(labelImage,3) == 3
        labelImage = rgb2gray(labelImage);
    end
    
    if iscategorical(labelImage)
        labelImageCat = labelImage;
    else
        labelIndices = zeros(size(labelImage), 'uint8');
        labelIndices(labelImage == 0) = 1;     % background
        labelIndices(labelImage == 255) = 2;   % person
        labelImageCat = categorical(labelIndices, [1 2], ["background", "person"]);
    end

    
    if ndims(labelImageCat) > 2
        labelImageCat = labelImageCat(:,:,1);
    end

 
    if size(inputImage,3) == 1
        inputImage = repmat(inputImage, [1 1 3]);
    end

 
    assert(isequal(size(inputImage,1:2), size(labelImageCat)), ...
        'Input image and label image size mismatch.');

    dataOut = {inputImage, labelImageCat};
end




% 3. 데이터 전처리
augmentedTrainingData = transform(trainingData, @(data)augmentData(data, imageSize));

% 4. 네트워크 생성 (DeepLabv3+ with resnet18)
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

% 5. 학습 옵션 설정
options = trainingOptions("adam", ...
    "InitialLearnRate",0.001, ...
    "MaxEpochs",20, ...
    "MiniBatchSize",4, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false);

% 6. 네트워크 학습
net = trainNetwork(augmentedTrainingData, lgraph, options);

% 7. 테스트 및 결과 시각화
testImage = imread("C:\Users\Owner\Desktop\matlab_contest\bikeperson.png");
testImage = imresize(testImage, imageSize(1:2));

% 예측 마스크
predictedMask = semanticseg(testImage, net);

% 마스크를 논리형으로 변환
personMask = predictedMask == "person";

% 1. 세그멘테이션 결과 시각화
figure;
subplot(2,3,1);
imshow(testImage);
title("Original Image");

subplot(2,3,2);
B = labeloverlay(testImage, predictedMask, "Transparency", 0.4);
imshow(B);
title("Segmentation Overlay");

% 2. 사람만 분리
subplot(2,3,3);
personOnly = testImage;
for c = 1:3
    channel = personOnly(:,:,c);
    channel(~personMask) = 0;
    personOnly(:,:,c) = channel;
end
imshow(personOnly);
title("Person Only");

% 3. 배경만 분리 (사람 제거)
subplot(2,3,4);
backgroundOnly = testImage;
for c = 1:3
    channel = backgroundOnly(:,:,c);
    channel(personMask) = 0;
    backgroundOnly(:,:,c) = channel;
end
imshow(backgroundOnly);
title("Background Only");

% 4. 사람 제거 후 채우기 (inpainting)
% RGB 각 채널별로 채움
subplot(2,3,5);
inpaintedImage = zeros(size(testImage), 'uint8');
for c = 1:3
    inpaintedImage(:,:,c) = regionfill(testImage(:,:,c), personMask);
end
imshow(inpaintedImage);
title("Inpainted Background");



