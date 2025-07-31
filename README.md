# MatlabChallenge2025
MATLABAIChallenge2025
포항공과대학교 무은재학부 김은채,김준희,안도윤

segmentationByResnet.m: resnet을 전이학습시킨 신경망을 이용해 배경과 사람 분리
picture_seg.py: 배경과 사람을 분리하는 파이썬 코드
repainting_func.m: 사람이 분리되어 빈 부분을 채워넣기
main_efficientb0.m: efficientnet을 전이학습시켜 depthmap 생성
main_mobile2.m: mobilenet을 전이학습시켜 depthmap 생성
main_squeezenet.m: squeezenet을 전이학습시켜 depthmap 생성
til_3D_trans.m: 배경 RGB-D로 pointcloud를 만들고 mesh로 전환
face.m: 사람 RGB-D로 pointcloud를 만들고 mesh로 전환
trimming.m: 배경 3D(obj 파일)을 직육면체 모양으로 다듬기
byeong.m: 사람과 배경 3D(obj 파일) 병합
