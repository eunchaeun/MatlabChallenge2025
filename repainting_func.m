function result = criminisi_inpaint(inputImage, mask, patchSize)
    % Criminisi-style exemplar-based inpainting
    % inputImage: RGB 이미지 (double [0~1])
    % mask: 1이면 비어 있음, 0이면 정상
    % patchSize: 정사각형 패치 크기

    if nargin < 3
        patchSize = 9;
    end

    img = im2double(inputImage);
    result = img;
    [H, W, ~] = size(img);

    % Confidence map (초기: 알려진 곳은 1, 나머지 0)
    C = double(~mask);

    halfPatch = floor(patchSize / 2);

    while any(mask(:))
        % 1. Fill front (경계)
        fillFront = bwperim(mask);

        % 2. 우선순위 계산
        [Iy, Ix] = find(fillFront);
        priorities = zeros(length(Iy), 1);

        for i = 1:length(Iy)
            y = Iy(i); x = Ix(i);

            % Patch 범위
            x1 = max(1, x - halfPatch); x2 = min(W, x + halfPatch);
            y1 = max(1, y - halfPatch); y2 = min(H, y + halfPatch);
            patchMask = mask(y1:y2, x1:x2);
            patchConf = C(y1:y2, x1:x2);

            % Confidence term
            C_p = sum(patchConf(~patchMask)) / numel(patchConf);

            % Data term 생략
            priorities(i) = C_p;
        end

        % 3. 가장 우선순위 높은 점 선택
        [~, maxIdx] = max(priorities);
        px = Ix(maxIdx);
        py = Iy(maxIdx);

        % 4. 패치 범위 설정
        x1 = max(1, px - halfPatch); x2 = min(W, px + halfPatch);
        y1 = max(1, py - halfPatch); y2 = min(H, py + halfPatch);

        targetPatch = result(y1:y2, x1:x2, :);
        targetMask  = mask(y1:y2, x1:x2);

        % 5. 가장 비슷한 패치 검색
        bestPatch = find_best_patch_fast(result, mask, targetPatch, targetMask, patchSize, px, py); % 수정했어요!!!!

        % 6. 복사 (비어있는 부분만)
        for ch = 1:3
            temp = targetPatch(:,:,ch);
            patchCh = bestPatch(:,:,ch);
            temp(targetMask) = patchCh(targetMask);
            result(y1:y2, x1:x2, ch) = temp;
        end

        % 7. Confidence 업데이트
        C(y1:y2, x1:x2) = max(C(y1:y2, x1:x2), ~targetMask);

        % 8. 마스크 업데이트
        mask(y1:y2, x1:x2) = mask(y1:y2, x1:x2) & ~targetMask;
    end

    disp("Criminisi Inpainting 완료!");
end


function bestPatch = find_best_patch_fast(img, mask, targetPatch, targetMask, patchSize, centerX, centerY)
    [H, W, ~] = size(img);
    halfPatch = floor(patchSize / 2);
    bestError = Inf;
    searchRadius = 40;

    [tH, tW, ~] = size(targetPatch);  % targetPatch 크기 저장
    found = false;

    for y = max(1+halfPatch, centerY-searchRadius):min(H-halfPatch, centerY+searchRadius)
        for x = max(1+halfPatch, centerX-searchRadius):min(W-halfPatch, centerX+searchRadius)
            patch = img(y-halfPatch:y+halfPatch, x-halfPatch:x+halfPatch, :);
            patchM = mask(y-halfPatch:y+halfPatch, x-halfPatch:x+halfPatch);

            % 크기 다르면 스킵 (가장자리 문제 회피)
            if size(patch,1) ~= tH || size(patch,2) ~= tW
                continue;
            end

            if any(patchM(:))
                continue;
            end

            % 차이 계산
            diff = (patch - targetPatch).^2;
            diff(targetMask) = 0;
            error = sum(diff(:));

            if error < bestError
                bestError = error;
                bestPatch = patch;
                found = true;
            end
        end
    end

    if ~found
        warning('No valid patch found — using targetPatch as fallback');
        bestPatch = targetPatch;
    end
end


%% 사용 예시
img = imread("C:\Workspace\matlab-contest\final\remove.png");
mask = all(img == 0, 3); 

output = criminisi_inpaint(img, mask, 50);
imshow(output);
