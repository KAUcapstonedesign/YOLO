import cv2
import dlib
import numpy as np
import joblib

# 학습된 모델 로드 및 dlib의 얼굴 감지기, 랜드마크 예측기 초기화
model = joblib.load('face_landmarks_trained_model.pkl')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
cap.set(3, 256)
cap.set(4, 256)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환
    faces = detector(gray)  # 얼굴 감지

    for face in faces:
        landmarks = predictor(gray, face) # 랜드마크 예측, face는 얼굴 좌표
        features = []   # 랜드마크에서 x, y 좌표 추출 및 특성 벡터 생성
        for n in range(0, landmarks.num_parts): # 랜드마크의 모든 점에 대해 반복 처리
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            features.append(x)  # 특성 벡터에 추가
            features.append(y)  # 특성 벡터에 추가
        
        features = np.array(features).reshape(1, -1) # 모델 입력 형태에 맞게 조정, 2차원 배열로 변환
        prediction = model.predict(features)    # 모델로 예측
        
        # 감지한 얼굴에 사각형 및 예측 결과에 따라 webcam에 이름 표시
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, prediction[0], (face.left(), face.top() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)  
    if cv2.waitKey(1) == 27:   
        break   

cap.release()
cv2.destroyAllWindows()