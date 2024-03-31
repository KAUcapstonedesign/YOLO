# **이미지에서 얼굴 인식 및 랜드마크 부위별 표시**
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

def show_raw_detection(image, detector, predictor): # 얼굴 인식 및 랜드마크 표시
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환

    rects = detector(gray, 1)   # 얼굴 인식 수행 (img, upsample_times)

    for (i, rect) in enumerate(rects):  # 인식된 얼굴 개수만큼 반복
        shape = predictor(gray, rect)   # 인식된 얼굴에서 랜드마크 찾기
        shape = face_utils.shape_to_np(shape)   # 랜드마크 좌표를 (x, y) 튜플로 변환

        (x, y, w, h) = face_utils.rect_to_bb(rect)  # 얼굴 윤곽을 표시하기 위한 좌표 계산
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0),2)  # 얼굴 윤곽을 초록색으로 표시
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)   # 얼굴 번호 표시
        for (x, y) in shape:    # 랜드마크 좌표를 빨간색으로 표시
            cv2.circle(image, (x,y), 1, (0, 0, 255), -1)    # 랜드마크 좌표를 빨간색으로 표시

    cv2.imshow("Output", image) # 얼굴 인식 및 랜드마크 표시된 이미지 출력
    cv2.waitKey(0)  # 키 입력 대기

def draw_individual_detections(image, detector, predictor):   # 각 랜드마크 별로 표시
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 흑백으로 변환
    rects = detector(gray, 1)   # 얼굴 인식 수행 (img, upsample_times)
    for (i, rect) in enumerate(rects):  # 인식된 얼굴 개수만큼 반복
        shape = predictor(gray, rect)   # 인식된 얼굴에서 랜드마크 찾기
        shape = face_utils.shape_to_np(shape)   # 랜드마크 좌표를 (x, y) 튜플로 변환
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():  # 랜드마크 별로 반복
            clone = image.copy()    # 이미지 복사
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)   # 랜드마크 이름 표시
            for (x, y) in shape[i:j]:   # 랜드마크 좌표를 빨간색으로 표시
                cv2.circle(clone, (x,y), 1, (0, 0, 255), -1)    # 랜드마크 좌표를 빨간색으로 표시
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))  # 랜드마크 좌표를 감싸는 사각형 계산
            roi = image[y:y + h, x:x + w]   # 랜드마크 좌표를 감싸는 사각형 영역 추출
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)  # 영역 크기 조정
            cv2.imshow("ROI", roi)  # 영역 출력
            cv2.imshow("Image", clone)  # 랜드마크 표시된 이미지 출력
            cv2.waitKey(0)  # 키 입력 대기

        output = face_utils.visualize_facial_landmarks(image,shape) # 랜드마크를 이미지에 표시
        cv2.imshow("Image", output) # 랜드마크 표시된 이미지 출력
        cv2.waitKey(0)  # 키 입력 대기

detector = dlib.get_frontal_face_detector() # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
predictor = dlib.shape_predictor('C:\capstonedesign\shape_predictor_68_face_landmarks.dat')  # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성

image = cv2.imread('C:\capstonedesign\me.jpg')   # 이미지 불러오기
image = imutils.resize(image, width=500)    # 이미지 크기 조정
show_raw_detection(image, detector, predictor)  # 얼굴 인식 및 랜드마크 표시
draw_individual_detections(image, detector, predictor)  # 각 랜드마크 별로 표시