import cv2
import dlib
import numpy as np
import json
import os

# 얼굴 인식 및 랜드마크 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# 원본 이미지에서 128차원 특징 벡터 추출
def extract_feature_vector(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_rgb, 1)
    for det in dets:
        shape = predictor(img_rgb, det)
        face_descriptor = face_rec_model.compute_face_descriptor(img_rgb, shape)
        return [f for f in face_descriptor]
    return None

# 웹캠 설정
capture = cv2.VideoCapture(0)
capture.set(3, 640)  # 카메라 너비 설정
capture.set(4, 480)  # 카메라 높이 설정

face_id = input('\n enter user id and press <enter> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0

# 원본 이미지의 특징 벡터 로드
original_face_vector = extract_feature_vector("C:\part divide\kicheol.jpg")

if original_face_vector is None:
    print("No face found in the original image.")
else:
    while True:
        ret, frame = capture.read()
        if not ret:
            print("[ERROR] Unable to capture video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        for face in faces:
            shape = predictor(gray, face)
            face_chip = dlib.get_face_chip(frame, shape, size=256, padding=0.3)
            
            # 정렬된 얼굴에서 랜드마크 검출 및 특징 벡터 추출
            aligned_shape = predictor(cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB), dlib.rectangle(0, 0, face_chip.shape[1], face_chip.shape[0]))
            face_descriptor = face_rec_model.compute_face_descriptor(face_chip, aligned_shape)
            current_face_vector = [f for f in face_descriptor]

            # 추출된 벡터와 원본 이미지의 벡터 비교
            distance = np.linalg.norm(np.array(original_face_vector) - np.array(current_face_vector))
            if distance < 0.4:
                print("Matched Face")
            else:
                print("Unmatched Face")

            count += 1
            # 정렬된 얼굴과 특징 벡터 저장
            img_path = f"dataset/face_{face_id}_{count}.jpg"
            cv2.imwrite(img_path, cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR))
            json_path = f"dataset/face_{face_id}_{count}_vector.json"
            with open(json_path, "w") as json_file:
                json.dump(current_face_vector, json_file)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == 27 or count >= 30:  # 'Esc'를 누르거나 100개의 얼굴이 캡처되면 종료
            break

    capture.release()
    cv2.destroyAllWindows()