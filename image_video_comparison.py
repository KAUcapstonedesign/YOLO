# **이미지에서 추출한 랜드마크와 실시간으로 캡처한 얼굴 랜드마크 간의 거리를 계산하여 얼굴 일치 여부 판단**
# GPT의 도움을 받아 작성한 코드
import cv2
import dlib
import numpy as np

# Dlib의 얼굴 검출기와 랜드마크 검출 모델 로드
detector = dlib.get_frontal_face_detector() # 얼굴 검출기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")   # 랜드마크 검출 모델

# 랜드마크를 numpy 배열로 변환하는 함수
def shape_to_np(shape, dtype="int"):    # shape: 랜드마크 좌표, dtype: 데이터 타입  
    coords = np.zeros((68, 2), dtype=dtype) # 68개의 랜드마크 좌표를 저장할 배열 생성
    for i in range(0, 68):  # 68개의 랜드마크 좌표를 (x, y) 형태로 변환
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords   # 변환한 랜드마크 좌표 배열 반환

# 두 랜드마크 집합 간의 거리를 계산하는 함수
def landmarks_distance(landmarks1, landmarks2):     # landmarks1: 첫 번째 랜드마크 집합, landmarks2: 두 번째 랜드마크 집합
    sum_distance = np.sum(np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=0)))      # 두 랜드마크 집합 간의 거리 계산
    print("Distance between landmarks:", sum_distance)  # 두 랜드마크 집합 간의 거리 출력
    return sum_distance    # 두 랜드마크 집합 간의 거리 반환

# 얼굴 랜드마크 추출 함수
def get_face_landmarks(image):  # image: 입력 이미지
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 입력 이미지를 흑백으로 변환
    faces = detector(gray, 1)   # 얼굴 검출
    
    if len(faces) > 0:  # 얼굴이 검출된 경우
        face = faces[0]  # 첫 번째 검출된 얼굴 사용
        landmarks = predictor(gray, face)   # 얼굴 랜드마크 검출
        landmarks = shape_to_np(landmarks)  # 랜드마크를 numpy 배열로 변환
        return landmarks    # 랜드마크 반환
    return None    # 얼굴이 검출되지 않은 경우 None 반환

# 두 얼굴 이미지 로드
image1 = cv2.imread("C:\shape_predictor_68_face_landmarks.dat/Face1.jpg")  
image2 = cv2.imread("C:\shape_predictor_68_face_landmarks.dat/Face2.jpg")

# 각 이미지에서 얼굴 랜드마크 추출
landmarks1 = get_face_landmarks(image1) 
print(landmarks1)
landmarks2 = get_face_landmarks(image2)
print(landmarks2)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
cap.set(3, 272) # 너비
cap.set(4, 272) # 높이

# 기준이 될 첫 번째 얼굴 랜드마크 설정
base_landmarks = None   # 초기화

# 임계값 설정
SOME_THRESHOLD = 8000  # 적절한 임계값으로 조정 필요

while True:
    ret, frame = cap.read() # 프레임 읽기
    if not ret: # 프레임을 읽지 못한 경우
        break
    
    # 현재 프레임에서 얼굴 랜드마크 추출
    landmarks = get_face_landmarks(frame)
    
    if landmarks is not None:   # 얼굴 랜드마크가 검출된 경우
        if base_landmarks is None:  # 기준 랜드마크가 설정되지 않은 경우
            base_landmarks = landmarks  # 현재 프레임의 얼굴 랜드마크를 기준 랜드마크로 설정
            print("Base face landmarks set.")
        else:
            # 기준 랜드마크와 현재 프레임의 얼굴 랜드마크 간의 거리 계산
            distance = landmarks_distance(base_landmarks, landmarks)
            
            # 거리에 따른 일치 여부 판단
            if distance < SOME_THRESHOLD:   # 거리가 임계값보다 작은 경우
                print("Faces match")
            else:
                print("Faces do not match")
    else:
        print("No face detected in the current frame.")
    
    cv2.imshow("Face", frame)   # 프레임 출력
    if cv2.waitKey(1) & 0xFF == 27: # ESC 키를 누른 경우
        break   # 종료

# 모든 자원 해제
cap.release()
cv2.destroyAllWindows()