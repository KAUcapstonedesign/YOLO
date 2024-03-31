# **이미지에서 얼굴 랜드마크 표시**
# # 터미널에 python face_landmark_detection.py shape_predictor_68_face_landmarks.dat C:\capstonedesign 입력하고 실행해야 함
import sys
import os
import dlib
import glob
import cv2 

# RGB > BGR or BGR > RGB 변환 
# dlib는 RGB 형태로 이미지를 사용하고 openCV는 BGR 형태이므로 B와 R을 바꿔주는 함수가 필요하다.
def swapRGB2BGR(rgb):   # rgb는 numpy 배열
    r, g, b = cv2.split(img)    # img파일을 b,g,r로 분리
    bgr = cv2.merge([b,g,r])    # b, r을 바꿔서 Merge
    return bgr                # bgr 결과 리턴

predictor_path = sys.argv[1]    # 첫 번째 매개변수: 68개의 얼굴 랜드마크가 학습된 모델 데이터
faces_folder_path = sys.argv[2] # 두 번째 매개변수: 랜드마크 적용할 이미지를 모아둔 폴더
# 세 번째 매개변수: 랜드마크 적용한 이미지를 저장할 폴더
detector = dlib.get_frontal_face_detector() # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
predictor = dlib.shape_predictor(predictor_path)    # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성
cv2.namedWindow('Face') # 이미지를 화면에 표시하기 위한 윈도 생성

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):   # 두 번째 매개변수로 지정한 폴더에서 jpg파일만 찾기
    print("Processing file: {}".format(f))  # 파일 경로 출력
    img = dlib.load_rgb_image(f)    # 파일을 img 변수에 저장(이미지 불러오기)
    
    cvImg = swapRGB2BGR(img)    # opencv용 BGR 이미지로 변환(R과 B를 바꿔줌)
    
    # 이미지 크기 작을 경우 두 배로 키우기
    # cvImg = cv2.resize(cvImg, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

    # 두번째 변수 1은 업샘플링을 한 번 하겠다는 얘기인데
    # 업샘플링을 하면 정확도가 올라가고, 얼굴을 더 많이 인식할 수 있다.
    # 다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
    # 그냥 1이면 될 듯.
    dets = detector(img, 1) # 얼굴 인식 수행 (img, upsample_times)
    print("Number of faces detected: {}".format(len(dets))) # 인식된 얼굴 개수 출력
    
    # 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽을 표시 
    for k, d in enumerate(dets):    # k 얼굴 인덱스, d 얼굴 좌표
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))   # 인식된 좌표(box d) 출력
        shape = predictor(img, d)   # 인식된 좌표에서 랜드마크 찾기
        print(shape.num_parts)  # 랜드마크 개수(num_parts) 출력 = 68개

        # num_parts(랜드마크 구조체)를 하나씩 0부터 67까지 루프를 돌린다
        for i in range(0, shape.num_parts):
            # 해당 X,Y 좌표를 두 배로 키워 좌표를 얻으려면 shape.part(i).x*2, shape.part(i).y*2를 하면 된다.
            x = shape.part(i).x
            y = shape.part(i).y
            print(str(x) + " " + str(y))    # 랜드마크 좌표값 출력
            # 이미지 랜드마크 좌표 지점에 인덱스(랜드마크 번호, 여기선 i)를 putText로 표시 
            cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))             
        cv2.imshow('Face', cvImg)   # 랜드마크가 표시된 이미지를 openCV 윈도에 표시

    # 무한으로 대기하다가 ESC를 누르면 빠져나와 다음 이미지를 검색
    while True:
        if cv2.waitKey(0) == 27:
            break;

cv2.destroyWindow('Face')   # openCV 윈도 제거