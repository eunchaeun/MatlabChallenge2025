import torch
import torchvision
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import resize
import cv2  

# 1. 모델 로드
weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights).eval()

# 2. 이미지 로드
image_path = "C:/Users/sec/Documents/MATLAB/humanpic.jpg"
input_image = Image.open(image_path).convert("RGB")
width, height = input_image.size

# 3. 전처리
preprocess = weights.transforms()
input_tensor = preprocess(input_image).unsqueeze(0)

# 4. 추론
with torch.no_grad():
    output = model(input_tensor)["out"][0]
output_predictions = output.argmax(0)  

# 5. 마스크 업샘플링
mask_resized = resize(output_predictions.unsqueeze(0).float(), [height, width], interpolation=0)[0].byte()

# 6. 사람 클래스만 선택 
mask = (mask_resized == 15)

# 7. 입력 이미지 numpy 배열화
input_np = np.array(input_image)

# 8. 사람만 남긴 이미지 저장
person_img = input_np.copy()
person_img[~mask.numpy()] = 0
Image.fromarray(person_img).save("person_segmented5.png")

# 9. 배경만 남긴 이미지 저장
background_img = input_np.copy()
background_img[mask.numpy()] = 0
Image.fromarray(background_img).save("background_segmented5.png")

# 10. 빈 공간(inpaint용 마스크) 생성
inpaint_mask = (mask.numpy().astype(np.uint8)) * 255  

# OpenCV에 맞게 RGB -> BGR 변환
background_bgr = cv2.cvtColor(background_img, cv2.COLOR_RGB2BGR)

# 11. Inpainting 수행
inpainted = cv2.inpaint(background_bgr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# BGR -> RGB 복원 후 저장
inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
Image.fromarray(inpainted_rgb).save("background_inpainted5.png")
