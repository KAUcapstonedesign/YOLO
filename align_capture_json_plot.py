# **얼굴 정렬부터 하고 찍기, 저장, 랜드마크 추출하여 JSON 파일로 저장, 각 점들 분포도 출력**
import cv2
import dlib
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# 모델 및 카메라 초기화
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
capture = cv2.VideoCapture(0)
capture.set(3, 256)
capture.set(4, 256)

face_id = input('\n enter user id end press <enter> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

# 얼굴 정렬 함수
def align_faces(frame):
    dets = detector(frame, 1)
    faces = []
    for det in dets:
        shape = predictor(frame, det)
        face_chip = dlib.get_face_chip(frame, shape, size=256, padding=0.3)
        faces.append(face_chip)
    return faces

json_files = []  # JSON 파일 경로를 저장할 리스트

# 얼굴 캡처 및 랜드마크 추출
while True:
    ret, frame = capture.read()
    if not ret:
        print("[ERROR] Unable to capture video")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.05, 3, minSize=(20,20))
    
    # 얼굴 정렬 및 저장
    if len(faces) > 0:
        count += 1
        aligned_faces = align_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        for aligned_face in aligned_faces:
            aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
            img_path = f"dataset/face_{face_id}_{count}.jpg"
            cv2.imwrite(img_path, aligned_gray)
            
            # 랜드마크 추출 및 JSON 파일 저장
            rects = detector(aligned_gray, 1)
            for rect in rects:
                landmarks = predictor(aligned_gray, rect)
                landmark_list = [[p.x, p.y] for p in landmarks.parts()]

                json_path = f"dataset/face_{face_id}_{count}.json"
                json_files.append(json_path)  # JSON 파일 경로 저장
                with open(json_path, "w") as json_file:
                    json.dump(landmark_list, json_file)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27 or count >= 100:  # ESC 키를 누르거나 100회 캡처 시 종료
        break

print("\n [INFO] Exiting Program and cleanup stuff")
capture.release()
cv2.destroyAllWindows()

# 랜드마크 데이터 로드 및 시각화
plt.figure(figsize=(10, 10))  # 그래프 크기 설정

for json_path in json_files:
    with open(json_path) as file:
        data = json.load(file)
        x, y = zip(*data)  # x, y 좌표 분리
        plt.scatter(x, y, alpha=0.5)  # 랜드마크 시각화

plt.gca().invert_yaxis()  # y축을 상단이 0이 되도록 변경
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Facial Landmarks')
plt.show()
