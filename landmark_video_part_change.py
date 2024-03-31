# **실시간 얼굴 인식 및 키보드로 랜드마크 선택**
import dlib
import cv2 as cv
import numpy as np

detector = dlib.get_frontal_face_detector() # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성

cap = cv.VideoCapture(0)    # 비디오 캡쳐 객체 생성

# range는 끝값이 포함 안 됨
ALL = list(range(0, 68))  # 랜드마크 인덱스
JAWLINE = list(range(0, 17))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))

index = ALL # 랜드마크 인덱스 초기값

while True:

    ret, img_frame = cap.read() # 비디오 프레임 읽기

    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)    # 이미지를 흑백으로 변환

    dets = detector(img_gray, 1)    # 얼굴 인식 수행 (img, upsample_times)

    for face in dets:   # 인식된 얼굴 개수만큼 반복

        shape = predictor(img_frame, face)  # 인식된 얼굴에서 랜드마크 68개 찾기

        list_points = []    # 랜드마크 좌표를 저장할 리스트
        for p in shape.parts(): # 랜드마크 좌표를 (x, y) 튜플로 변환
            list_points.append([p.x, p.y])  # 랜드마크 좌표를 리스트에 저장
        list_points = np.array(list_points) # 랜드마크 좌표를 numpy 배열로 변환

        for i,pt in enumerate(list_points[index]):  # 랜드마크 인덱스에 따라 반복
            pt_pos = (pt[0], pt[1]) # 랜드마크 좌표
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)    # 랜드마크 좌표를 초록색으로 표시
        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)   # 얼굴 윤곽을 빨간색으로 표시

    cv.imshow('result', img_frame)  # 랜드마크 표시된 이미지 출력

    key = cv.waitKey(1) # 키 입력 대기

    if key == 27:   # ESC 키 입력시 종료
        break
    elif key == ord('1'):   # 1번 키 입력시 전체 랜드마크 표시
        index = ALL
    elif key == ord('2'):   # 2번 키 입력시 눈썹 랜드마크 표시
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):   # 3번 키 입력시 눈 랜드마크 표시
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):   # 4번 키 입력시 코 랜드마크 표시
        index = NOSE
    elif key == ord('5'):   # 5번 키 입력시 입 랜드마크 표시
        index = MOUTH_OUTLINE+MOUTH_INNER
    elif key == ord('6'):   # 6번 키 입력시 턱 랜드마크 표시
        index = JAWLINE

cap.release()   # 비디오 캡쳐 객체 해제