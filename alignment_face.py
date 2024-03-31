# **이미지에서 얼굴 정렬 -> 잘 안됨**
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector() # 얼굴 인식 모델
model = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')           # 랜드마크 모델
img = dlib.load_rgb_image('side.jpg')   # 이미지 불러오기
objs = dlib.full_object_detections()    # 객체 생성 후 랜드마크 저장
dets = detector(img, 1) # 얼굴 인식 후 바운딩 박스 그리기

for detection in dets:  # 얼굴 인식 후 랜드마크 찾기
    s = model(img, detection)   # 랜드마크 찾기
    objs.append(s)  # 랜드마크 저장

faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)   # 얼굴 정렬(size는 결과 이미지 크기, padding은 얼굴 주변 여백)

# 이미지 출력을 위한 준비
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))
axes[0].imshow(img) # 좌측에 원본 이미지 출력
for i, face in enumerate(faces):
    axes[i+1].imshow(face)  # 우측에 얼굴 정렬 이미지 출력

# 정렬된 얼굴 이미지 저장
for i, face in enumerate(faces):
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)    # RGB 이미지를 BGR로 변환 (OpenCV는 BGR 포맷을 사용)
    save_path = f'C:/capstonedesign/side_aligned.jpg'  # 저장할 경로와 파일명
    cv2.imwrite(save_path, face_bgr)

plt.show()  # 결과 창 띄우기

'''
# 모델 불러오기
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

# 이미지 불러오기
img = dlib.load_rgb_image('side.jpg')
plt.imshow(img)

# detector로 얼굴 인식 후 바운딩 박스 그리기
dets = detector(img, 1)
fig, ax = plt.subplots(1, figsize=(16, 10))
for det in dets:
    x, y, w, h = det.left(), det.top(), det.width(), det.height()
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax.imshow(img)

# 5 landmark model을 이용해 눈, 코, 입 위치 찾기
fig, ax = plt.subplots(1, figsize=(16, 10))
objs = dlib.full_object_detections()
for detection in dets:
    s = model(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=2, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img)

# 발견된 위치와 이미지로 face-align 수행
faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20, 16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
'''