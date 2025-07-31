%  1. 테스트 이미지 준비 
rgbFile = "C:\Users\sec\Documents\MATLAB\human_backgroundX.png";
depthFile = "C:\Users\sec\Documents\MATLAB\human_backgroundX_depth.png";
testRGB = imread(rgbFile);
predictedDepth_raw = imread(depthFile);

% 네트워크 입력 규격(227×227)에 맞추기
netInputSize = [227 227];

% 원본 이미지 크기 확인
fprintf('=== 원본 이미지 정보 ===\n');
fprintf('RGB 원본 크기: %s\n', mat2str(size(testRGB)));
fprintf('깊이 원본 크기: %s\n', mat2str(size(predictedDepth_raw)));

testRGB_resized = imresize(testRGB, netInputSize);
predictedDepth_resized = imresize(predictedDepth_raw, netInputSize);

% 리사이즈 후 크기 확인
fprintf('=== 리사이즈 후 정보 ===\n');
fprintf('RGB 리사이즈 후: %s\n', mat2str(size(testRGB_resized)));
fprintf('깊이 리사이즈 후: %s\n', mat2str(size(predictedDepth_resized)));

% 깊이 이미지가 3차원이면 첫 번째 채널만 사용
if ndims(predictedDepth_resized) == 3
    predictedDepth_resized = predictedDepth_resized(:,:,1);
    fprintf('깊이 이미지를 2D로 변환: %s\n', mat2str(size(predictedDepth_resized)));
end

% Depth 값을 double로 변환 후 [m] 단위 스케일
Z = double(predictedDepth_resized)/1000; % mm → m

% 2. 포인트 클라우드 만들기 
% ─ 카메라 내참값(가정) 
fx = 200; fy = 200;
cx = (netInputSize(2)+1)/2; % =113.5
cy = (netInputSize(1)+1)/2;

% ─ 픽셀 → 3-D 좌표 변환 
[H,W] = size(Z); % 깊이 이미지 크기 사용
[U,V] = meshgrid(1:W, 1:H); % 픽셀 좌표
X3D = (U - cx) .* Z / fx;
Y3D = (V - cy) .* Z / fy;

% ─ 크기 확인 및 조정 
fprintf('=== 처리 전 최종 확인 ===\n');
fprintf('RGB 이미지 크기: %s\n', mat2str(size(testRGB_resized)));
fprintf('깊이 이미지 크기: %s\n', mat2str(size(Z)));

% 크기가 정확히 일치하는지 확인
assert(size(testRGB_resized,1) == size(Z,1) && size(testRGB_resized,2) == size(Z,2), ...
    'RGB와 깊이 이미지의 크기가 일치하지 않습니다!');

% ─ 유효한 포인트만 선택 (NaN이 아닌 깊이값) 
valid_mask = ~isnan(Z) & Z > 0;

% 모든 데이터를 1차원 배열로 변환
X3D_flat = X3D(:);
Y3D_flat = Y3D(:);
Z_flat = Z(:);
valid_mask_flat = valid_mask(:);

% 색상 데이터 준비
colors = reshape(testRGB_resized, [], 3); % uint8, 0-255

% 크기 확인
fprintf('=== 배열 크기 확인 ===\n');
fprintf('전체 픽셀 수: %d\n', length(X3D_flat));
fprintf('색상 데이터 행 수: %d\n', size(colors,1));
fprintf('유효한 포인트 수: %d\n', sum(valid_mask_flat));
fprintf('유효 비율: %.1f%%\n', 100*sum(valid_mask_flat)/length(valid_mask_flat));

% 유효한 포인트만 추출
points3D = [X3D_flat(valid_mask_flat) Y3D_flat(valid_mask_flat) Z_flat(valid_mask_flat)];
colors = colors(valid_mask_flat, :); % 유효한 포인트에 해당하는 색상만 선택

% pointCloud 객체
ptCloudColor = pointCloud(points3D, 'Color', colors);

%  3. 컬러 포인트 클라우드 시각화
figure('Name','Colored Point Cloud');
pcshow(ptCloudColor, 'VerticalAxis','y','MarkerSize',30);
title('RGB-D Point Cloud'); 
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3); axis tight; camorbit(25,0);

% 4. 삼각 메시 생성 
% 4-1) 유효한 픽셀 좌표로 Delaunay 삼각분할
[U_flat, V_flat] = meshgrid(1:W, 1:H);
U_valid = U_flat(valid_mask);
V_valid = V_flat(valid_mask);
try
    tri = delaunay(U_valid, V_valid); % 유효한 픽셀 좌표로 삼각분할
    
    % 4-2) Triangulation 객체
    TR = triangulation(tri, points3D);
    
    % 4-3) 정규화된 정점 컬러 (0-1)
    vertexColors = double(colors)./255;
    
    % 4-4) 메시 시각화
    figure('Name','Textured 3-D Mesh');
    trisurf(TR, ...
        'FaceVertexCData', vertexColors, ... % RGB 정점 색
        'FaceColor', 'interp', ...
        'EdgeColor', 'none');
    axis equal off; view(3);
    lighting gouraud; camlight headlight;
    title('Reconstructed 3-D Mesh (vertex-colored)');
    
catch ME
    fprintf('메시 생성 중 오류 발생: %s\n', ME.message);
end

% 5. 추가 정보 출력 
fprintf('원본 이미지 크기: %d x %d\n', size(testRGB_resized,1), size(testRGB_resized,2));
fprintf('전체 픽셀 수: %d\n', H*W);
fprintf('유효한 포인트 수: %d (%.1f%%)\n', size(points3D,1), 100*size(points3D,1)/(H*W));
fprintf('깊이 범위: %.3f ~ %.3f m\n', min(Z(valid_mask)), max(Z(valid_mask)));

% 6. 향상된 메시 생성 및 저장 
fprintf('\n=== 향상된 메시 생성 ===\n');

% 6-1) 깊이 맵 후처리
Z_filtered = medfilt2(Z, [3 3]);      % median filter로 노이즈 제거

% 6-2) 유효 깊이 마스크 (배경 제거)
validMask = (Z_filtered > 0.1) & (Z_filtered < 3.0);  % 거리 범위 지정
Z_filtered(~validMask) = NaN;               % 배경 제거

% 6-3) 픽셀 좌표 → 3D 변환 (필터링된 깊이 사용)
X_filtered = (U - cx) .* Z_filtered / fx;
Y_filtered = (V - cy) .* Z_filtered / fy;

% 6-4) 유효한 포인트만 추출
valid_indices = find(~isnan(Z_filtered));
if length(valid_indices) < 3
    fprintf('유효한 포인트가 너무 적습니다 (%d개)\n', length(valid_indices));
    return;
end

vertices = [X_filtered(valid_indices), Y_filtered(valid_indices), Z_filtered(valid_indices)];
vertex_colors = reshape(im2double(testRGB_resized), [], 3);
vertex_colors = vertex_colors(valid_indices, :);

% 6-5) 2D 픽셀 좌표에서 들로네 삼각분할
[rows, cols] = ind2sub(size(Z_filtered), valid_indices);
try
    faces = delaunay(cols, rows);
    
    % 6-6) triangulation 객체 생성
    TR_filtered = triangulation(faces, vertices);
    
    % 6-7) STL 저장 (triangulation 객체 사용)
    stlwrite(TR_filtered, 'reconstructed_filtered_mesh.stl');
    fprintf('STL 파일 저장 완료: reconstructed_filtered_mesh.stl\n');
    
    % 6-8) OBJ 저장 (색상 포함)
    objFile = fopen('reconstructed_filtered_mesh.obj','w');
    % 정점 쓰기 (위치 + 색상)
    for i = 1:size(vertices,1)
        fprintf(objFile, 'v %f %f %f %f %f %f\n', ...
            vertices(i,1), vertices(i,2), vertices(i,3), ...
            vertex_colors(i,1), vertex_colors(i,2), vertex_colors(i,3));
    end
    % 면 쓰기
    for i = 1:size(faces,1)
        fprintf(objFile, 'f %d %d %d\n', faces(i,1), faces(i,2), faces(i,3));
    end
    fclose(objFile);
    fprintf('OBJ 파일 저장 완료: reconstructed_filtered_mesh.obj\n');
    
    % 6-9) 시각화
    figure('Name','Filtered 3D Mesh','Color','w');
    trisurf(TR_filtered, 'FaceVertexCData', vertex_colors, ...
        'FaceColor', 'interp', 'EdgeColor', 'none');
    axis equal;
    view(3);
    camlight;
    lighting gouraud;
    title('Filtered & Colored 3D Mesh');
    rotate3d on; 
    
    fprintf('메시 생성 완료: 정점 %d개, 면 %d개\n', size(vertices,1), size(faces,1));
    
catch ME
    fprintf('향상된 메시 생성 중 오류 발생: %s\n', ME.message);
end



