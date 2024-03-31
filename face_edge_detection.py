# **이미지에서 얼굴 윤곽 표시**
# 터미널에 python face_edge_detection.py shape_predictor_68_face_landmarks.dat C:\capstonedesign 입력하고 실행해야 함
import sys
import os
import dlib  
import glob

predictor_path = sys.argv[1]    # 첫 번째 매개변수: 68개의 얼굴 랜드마크가 학습된 모델 데이터
faces_folder_path = sys.argv[2] # 두 번째 매개변수: 랜드마크 적용할 이미지를 모아둔 폴더
# 세 번째 매개변수: 랜드마크 적용한 이미지를 저장할 폴더
detector = dlib.get_frontal_face_detector()  # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
predictor = dlib.shape_predictor(predictor_path)    # 인식된 얼굴에서 랜드마크 찾기 위한 클래스 생성
win = dlib.image_window()   # 이미지를 화면에 표시하기 위한 윈도 생성

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):   # 두 번째 매개변수로 지정한 폴더에서 jpg 파일만 찾기
    print("Processing file: {}".format(f))  # 파일 경로 출력
    img = dlib.load_rgb_image(f)    # 파일을 img 변수에 저장(이미지 불러오기)

    win.clear_overlay() # 윈도 화면 지우기
    win.set_image(img)  # 현재 이미지를 화면에 출력

    # 두번째 변수 1은 업샘플링을 한 번 하겠다는 얘기인데
    # 업샘플링을 하면 정확도가 올라가고, 얼굴을 더 많이 인식할 수 있다.
    # 다만 값이 커질수록 느리고 메모리도 많이 잡아먹는다.
    # 그냥 1이면 될 듯.
    dets = detector(img, 1) # 얼굴 인식 수행 (img, upsample_times)
    print("Number of faces detected: {}".format(len(dets))) # 인식된 얼굴 개수 출력

    # 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽 표시
    for k, d in enumerate(dets):    # k 얼굴 인덱스, d 얼굴 좌표
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))   # 인식된 좌표(box d) 출력
        shape = predictor(img, d)   # 인식된 좌표(box d)에서 랜드마크 찾기

        # part(0)부터 part(67)까지 총 68개의 X,Y 좌표를 가지고 있다.
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))   # 랜드마크 중 처음 2개 출력
        win.add_overlay(shape)  # 랜드마크 화면 출력

    win.add_overlay(dets)   # 얼굴 윤곽 화면 출력
    dlib.hit_enter_to_continue()    # 다음 이미지로 넘어가기 위해 엔터키 입력