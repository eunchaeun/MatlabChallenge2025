function test()

% 1. 경로 설정
rgbPath   = 'rgbfile.png';
depthPath = 'depthfile';
outputOBJ = 'outputfile';

% 2. 이미지 불러오기
rgb = im2double(imread(rgbPath));
depth = double(imread(depthPath));
[H, W] = size(depth);

% 3. 메쉬 해상도 줄이기 (속도 개선)
step = 4;
[xGrid, yGrid] = meshgrid(1:step:W, 1:step:H);
z = depth(1:step:end, 1:step:end);
colors = rgb(1:step:end, 1:step:end, :);

% 4. 3D 점 구성
X = xGrid(:);
Y = yGrid(:);
Z = z(:);
C = reshape(colors, [], 3);

% 유효 깊이 필터링
valid = Z > 0;
X = X(valid); Y = Y(valid); Z = Z(valid); C = C(valid, :);

% 5. Delaunay 삼각화
tri = delaunay(X, Y);

% 외곽 직육면체 생성
minX = min(X); maxX = max(X);
minY = min(Y); maxY = max(Y);
minZ = min(Z); maxZ = max(Z);
pad = 5;
zExtend = 5;

% 직육면체 꼭짓점
boxCorners = [
    minX-pad, minY-pad, minZ-pad;
    maxX+pad, minY-pad, minZ-pad;
    maxX+pad, maxY+pad, minZ-pad;
    minX-pad, maxY+pad, minZ-pad;
    minX-pad, minY-pad, maxZ+pad;
    maxX+pad, minY-pad, maxZ+pad;
    maxX+pad, maxY+pad, maxZ+pad;
    minX-pad, maxY+pad, maxZ+pad
];

% 직육면체 면을 삼각형으로 정의 
boxTri = [
    1 2 3; 1 3 4;  % 아래면
    5 6 7; 5 7 8;  % 윗면
    2 3 7; 2 7 6;  % 오른면
    3 4 8; 3 8 7;  % 뒷면
    4 1 5; 4 5 8;  % 왼면
];
boxColor = repmat([0.8 0.8 0.8], 8, 1); % 연회색

% 7. 병합
V = [X Y Z; boxCorners];
F = [tri; boxTri + size(X,1)];
C = [C; boxColor];

% 8. OBJ 저장
writeOBJ(outputOBJ, V, F, C);

fprintf('저장 완료: %s\n', outputOBJ);

end

% -------------------------------
function writeOBJ(filename, vertices, faces, colors)
fid = fopen(filename, 'w');
fprintf(fid, '# OBJ 파일 생성\n');

% Vertex + 색상
for i = 1:size(vertices,1)
    fprintf(fid, 'v %.4f %.4f %.4f %.4f %.4f %.4f\n', ...
        vertices(i,1), vertices(i,2), vertices(i,3), ...
        colors(i,1), colors(i,2), colors(i,3));
end

% Face (1-indexed)
for i = 1:size(faces,1)
    fprintf(fid, 'f %d %d %d\n', faces(i,1), faces(i,2), faces(i,3));
end

fclose(fid);
end
