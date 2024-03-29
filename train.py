import cv2  # opencv 사용
import numpy as np # 배열 계산 용이
from PIL import Image # python imaging library
import os   # 운영체제 기능 사용

path = 'dataset' # 경로 (dataset 폴더 미리 생성)
recognizer = cv2.face.LBPHFaceRecognizer_create()   # LBPH 알고리즘 사용
detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')    # 얼굴 인식

def getImagesAndLabels(path):   # 얼굴 사진과 id 라벨을 가져오는 함수 선언
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   # path = dataset 폴더, f = 파일 이름
    # join(path,f) = path + file name (파일 경로 설정)
    # os.listdir : 해당 폴더 내 파일 리스트

    faceSamples = []    # 얼굴 사진
    ids = []    # id 라벨
    for imagePath in imagePaths:    # 각 파일마다 grayscale로 변환
        PIL_img = Image.open(imagePath).convert('L') # 이미지를 열고 'L = 8 bit pixel'로 바꿔줌(0~255의 수로 표현 가능한 grayscale 이미지 생성)
        img_numpy = np.array(PIL_img, 'uint8')  # 픽셀 처리를 위해 numpy 배열로 변환

        id = int(os.path.split(imagePath)[-1].split("_")[1])
        # Face_0_99.jpg 형식으로 저장된 파일을 처리하기 위해 파일 이름과 확장자명 분리 후 첫 번째 id(0번째 user) 값을 가져옴

        faces = detector.detectMultiScale(img_numpy)    # 얼굴 검출
        for(x,y,w,h) in faces:  # 얼굴이 검출되면
            faceSamples.append(img_numpy[y:y+h,x:x+w])  # 얼굴 부분만 저장
            ids.append(id)  # id 저장

    return faceSamples, ids # 얼굴 사진과 id 라벨 반환

print('\n [INFO] Training faces. Wait ...') # 진행 상황 알려주는 메시지
faces, ids = getImagesAndLabels(path)   # 얼굴 사진과 id 라벨 가져오기

recognizer.train(faces,np.array(ids)) # 학습

recognizer.write('trainer/trainer.yml') # 학습 결과 저장
print('\n [INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))   # 진행 상황 알려주는 메시지