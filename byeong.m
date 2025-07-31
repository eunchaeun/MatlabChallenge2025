function merge_two_obj_colored(humanFile, backgroundFile, outputFile, offset)
    % 두 개의 .obj 파일을 읽어서 병합하되,
    % 사람을 배경 앞쪽으로 yOffset만큼 이동하고,
    % vertex color도 유지

    % --- 사람 obj 읽기 ---
    [V1, F1] = readObjWithColor(humanFile);

    % --- 배경 obj 읽기 ---
    [V2, F2] = readObjWithColor(backgroundFile);

    V2(:, [2 3]) = [V2(:,3), -V2(:,2)];
    V1(:,2) = -V1(:,2);
    V1(:,2) = V1(:,2) - min(V1(:,2));

    % --- 크기 줄이기 ---
    scale = 0.8;
    V1(:,1:3) = V1(:,1:3) * scale;

    % --- x, y, z 오프셋 적용 ---
    V1(:,1) = V1(:,1) + offset(1);  % x
    V1(:,2) = V1(:,2) + offset(2);  % y
    V1(:,3) = V1(:,3) + offset(3);  % z


    % --- 병합 ---
    V_combined = [V1; V2];
    F2_shifted = F2 + size(V1, 1);
    F_combined = [F1; F2_shifted];

    % --- 저장 ---
    writeObjWithColor(outputFile, V_combined, F_combined);

    fprintf('✔️ Merged colored OBJ saved to: %s\n', outputFile);
end

function [V, F] = readObjWithColor(filename)
    % vertex (x y z [r g b]) & face 읽기
    fid = fopen(filename, 'r');
    V = [];
    F = [];
    while ~feof(fid)
        tline = strtrim(fgetl(fid));
        if startsWith(tline, 'v ')
            nums = sscanf(tline(3:end), '%f');
            % 길이에 따라 rgb 유무 판단
            if length(nums) == 3
                V = [V; nums(:)', 1, 1, 1]; % 기본 흰색
            elseif length(nums) == 6
                V = [V; nums(:)'];
            end
        elseif startsWith(tline, 'f ')
            nums = sscanf(tline(3:end), '%d');
            F = [F; nums(:)'];
        end
    end
    fclose(fid);
end






function writeObjWithColor(filename, V, F)
    % vertex with color + face 저장
    fid = fopen(filename, 'w');
    for i = 1:size(V,1)
        fprintf(fid, 'v %.6f %.6f %.6f %.6f %.6f %.6f\n', ...
            V(i,1), V(i,2), V(i,3), V(i,4), V(i,5), V(i,6));
    end
    for i = 1:size(F,1)
        fprintf(fid, 'f %d %d %d\n', F(i,1), F(i,2), F(i,3));
    end
    fclose(fid);
end

%%
merge_two_obj_colored( ...
    "C:\Workspace\matlab-contest\people_colored.obj", ...
    "C:\Workspace\matlab-contest\final\background.obj", ...
    'combined_fin.obj', ...
    [0, +30, -20]);  % 사람을 z축으로 10만큼 앞쪽에


