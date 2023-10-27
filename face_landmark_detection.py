import sys
import os
import dlib
import glob

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]    # 랜드마크 파일 경로
faces_folder_path = sys.argv[2] # 이미지 경로

detector = dlib.get_frontal_face_detector()  # 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
predictor = dlib.shape_predictor(predictor_path)    # 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성
win = dlib.image_window()   # 이미지를 화면에 표시하기 위한 윈도 생성

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):   # jpg 파일만 찾음
    print("Processing file: {}".format(f))  # 파일 경로 출력
    img = dlib.load_rgb_image(f)    # 파일을 img 변수에 저장

    win.clear_overlay() # 화면 지우기
    win.set_image(img)  # img를 화면에 출력

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1) # 1은 업샘플링 횟수
    print("Number of faces detected: {}".format(len(dets))) # 인식된 얼굴 개수 출력
    for k, d in enumerate(dets):        # 인식된 얼굴 개수만큼 반복하여 얼굴 윤곽 표시  
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format( 
            k, d.left(), d.top(), d.right(), d.bottom()))   # 인식된 좌표 출력
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)    # 인식된 좌표로부터 랜드마크 추출
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))   # 랜드마크 중 처음 2개 출력
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)  # 랜드마크 화면 출력

    win.add_overlay(dets)   # 얼굴 윤곽 화면 출력
    dlib.hit_enter_to_continue()    # 다음 이미지로 넘어가기 위해 엔터키 입력

#python face_landmark_detection.py shape_predictor_68_face_landmarks.dat ./
