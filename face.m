clc; clear;

% === 1. 이미지 및 알파 채널 로드 ===
[img, ~, alpha] = imread('C:\Workspace\matlab-contest\final\people.png');
img = imresize(img, [256 256]);
alpha = imresize(alpha, [256 256]);
mask = alpha > 0;

[h, w, ~] = size(img);
[X, Y] = meshgrid(1:w, 1:h);

% === 2. 깊이 맵 생성 ===
Z = double(mask) * 30;  % 앞면만 단일 깊이

% === 3. 유효한 픽셀만 추출 ===
X = X(mask);
Y = Y(mask);
Z = Z(mask);
colors = reshape(img, [], 3);
colors = colors(mask(:), :);

% === 4. 포인트 클라우드 객체 생성 ===
points = [X(:), Y(:), Z(:)];
ptCloud = pointCloud(points, 'Color', colors);

% === 5. .ply 파일로 저장 ===
pcwrite(ptCloud, 'people_colored.ply');

disp('색상 포함된 .ply 파일이 생성되었습니다:');
disp('   - 파일: people_colored.ply');


%%

clc; clear;

% === 1. PLY 불러오기 ===
ptCloud = pcread("C:\Workspace\matlab-contest\people_colored.ply");
points = ptCloud.Location;      % [N x 3]
colors = ptCloud.Color;         % [N x 3]

% === 2. NaN 제거 ===
valid = ~any(isnan(points), 2);  % NaN 있는 점 제외
points = points(valid, :);
colors = colors(valid, :);

% === 3. 2D 그리드 기반 삼각화 (가정: 원래 이미지 기반) ===
% 원래 이미지의 x, y 좌표로 Delaunay 삼각화
x = points(:,1);
y = points(:,2);
z = points(:,3);

tri = delaunay(x, y);  % 2D 평면에서 삼각망

% === 4. OBJ 파일로 저장 ===
fid = fopen('people_colored.obj', 'w');

% 정점 (v): 위치 + RGB
for i = 1:size(points,1)
    c = double(colors(i,:)) / 255;
    fprintf(fid, 'v %.4f %.4f %.4f %.4f %.4f %.4f\n', ...
        points(i,1), points(i,2), points(i,3), c(1), c(2), c(3));
end

% 면 (f): 삼각형 인덱스
for i = 1:size(tri,1)
    fprintf(fid, 'f %d %d %d\n', tri(i,1), tri(i,2), tri(i,3));
end

fclose(fid);
disp('PLY → OBJ 변환 완료: people_colored.obj');
